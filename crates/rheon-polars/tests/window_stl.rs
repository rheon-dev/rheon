//! `stl` and `mstl` through a window run: the formula form, the embargo's
//! defaults, each group's learned rows decomposed as `rheon_core` does on
//! the same window, a null learning nothing, any chunking and a save and
//! load between chunks giving the same bits, and the refusals by name.
use polars::prelude::*;
use rheon_core::{Mstl, Stl, StlCfg};
use rheon_polars::{WindowsConfig, WindowsRun};
use serde_json::json;

fn config(formulas: serde_json::Value) -> WindowsConfig {
    serde_json::from_value(
        json!({"formulas": formulas, "clock": "t", "gap_cap": 1e6, "group": "g"}),
    )
    .unwrap()
}

/// Two groups interleaved on a float clock, a null now and then.
fn frame(n: usize) -> DataFrame {
    let t: Vec<f64> = (0..n).map(|k| k as f64).collect();
    let g: Vec<&str> = (0..n).map(|k| if k % 3 == 0 { "a" } else { "b" }).collect();
    let x: Vec<Option<f64>> = (0..n)
        .map(|k| {
            (k % 13 != 5).then(|| {
                [1.5, -0.5, 2.0, -3.0][k % 4]
                    + [0.3, -0.1, 0.7, 0.2, -0.6, -0.5][k % 6]
                    + k as f64 * 0.05
            })
        })
        .collect();
    df!("t" => t, "g" => g, "x" => x).unwrap()
}

/// Feeds `df` in chunks of `size`, saving and loading after each.
fn run(config: &WindowsConfig, df: &DataFrame, size: usize) -> DataFrame {
    let schema = df.schema().as_ref().clone();
    let mut r = WindowsRun::new(config.clone(), &schema).unwrap();
    let mut parts = Vec::new();
    let mut at = 0;
    while at < df.height() {
        let chunk = df.slice(at as i64, size.min(df.height() - at));
        parts.push(r.feed(&chunk, None).unwrap());
        let bytes = r.save_bytes().unwrap();
        r = WindowsRun::load_bytes(&bytes, config.clone(), &schema).unwrap();
        assert_eq!(r.save_bytes().unwrap(), bytes, "a resave is the same bytes");
        at += size;
    }
    parts.push(r.finish().unwrap());
    let mut out = parts.remove(0);
    for p in parts {
        out.vstack_mut(&p).unwrap();
    }
    out
}

fn col(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
    let c = df.column(name).unwrap().f64().unwrap().clone();
    (0..c.len()).map(|i| c.get(i)).collect()
}

#[test]
fn each_row_is_its_groups_window_decomposed_at_the_lag() {
    let cfg = config(json!([
        {"name": "trend", "tree": ["stl", ["col", "x"], {"period": 4, "window_rows": 24, "lag": 5}]},
        {"name": "season", "tree": ["stl", ["col", "x"],
            {"period": 4, "window_rows": 24, "lag": 5, "closed": "right", "component": "seasonal"}]},
        {"name": "six", "tree": ["mstl", ["col", "x"],
            {"periods": [6, 4], "windows": [7, 7], "window_rows": 30, "lag": 6,
             "component": "seasonal", "season": 6}]},
    ]));
    let df = frame(150);
    let whole = run(&cfg, &df, 150);
    for size in [1, 7, 64] {
        assert!(
            run(&cfg, &df, size).equals_missing(&whole),
            "chunks of {size}"
        );
    }
    let stl = Stl::new(&StlCfg::new(4, 7)).unwrap();
    let mstl = Mstl::new(&[(6, 7), (4, 7)], &StlCfg::new(4, 7), 2).unwrap();
    let xs = col(&df, "x");
    let (trend, season, six) = (
        col(&whole, "trend"),
        col(&whole, "season"),
        col(&whole, "six"),
    );
    let mut learned: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for (k, x) in xs.iter().enumerate() {
        let gi = usize::from(k % 3 != 0);
        let before = learned[gi].clone();
        if let Some(x) = x {
            learned[gi].push(*x);
        }
        let after = &learned[gi];
        let left = (before.len() >= 24).then(|| stl.fit(&before[before.len() - 24..]).trend[19]);
        assert_eq!(
            trend[k].map(f64::to_bits),
            left.map(f64::to_bits),
            "row {k} trend"
        );
        let right = (after.len() >= 24).then(|| stl.fit(&after[after.len() - 24..]).seasonal[18]);
        assert_eq!(
            season[k].map(f64::to_bits),
            right.map(f64::to_bits),
            "row {k} seasonal"
        );
        let m =
            (before.len() >= 30).then(|| mstl.fit(&before[before.len() - 30..]).seasonal[1][24]);
        assert_eq!(
            six[k].map(f64::to_bits),
            m.map(f64::to_bits),
            "row {k} mstl"
        );
    }
}

#[test]
fn the_defaults_are_the_embargo_and_closed_left() {
    // period 4, span 7: h = 4 * 6 / 2 = 12, W = 25, lag 13 under "left".
    let cfg = config(json!([{"name": "s", "tree": ["stl", ["col", "x"], {"period": 4}]}]));
    let df = frame(80);
    let out = run(&cfg, &df, 80);
    let stl = Stl::new(&StlCfg::new(4, 7)).unwrap();
    let mut learned = Vec::new();
    for (k, x) in col(&df, "x").iter().enumerate() {
        let before = learned.clone();
        if k % 3 != 0
            && let Some(x) = x
        {
            learned.push(*x);
        }
        if k % 3 == 0 {
            continue;
        }
        let want = (before.len() >= 25).then(|| stl.fit(&before[before.len() - 25..]).trend[12]);
        assert_eq!(
            col(&out, "s")[k].map(f64::to_bits),
            want.map(f64::to_bits),
            "row {k}"
        );
    }
}

#[test]
fn refusals_name_the_operator() {
    let df = frame(10);
    let schema = df.schema().as_ref().clone();
    let refused = |tree: serde_json::Value| {
        // A parameter is refused as the config is read, a setting as the
        // run is built.
        let v = json!({"formulas": [{"name": "s", "tree": tree}], "clock": "t",
                       "gap_cap": 1e6, "group": "g"});
        match serde_json::from_value::<WindowsConfig>(v) {
            Err(e) => e.to_string(),
            Ok(cfg) => match WindowsRun::new(cfg, &schema) {
                Ok(_) => panic!("accepted"),
                Err(e) => e.to_string(),
            },
        }
    };
    let e = refused(json!(["stl", ["col", "x"], {"period": 4, "seasonal": 6}]));
    assert!(e.contains("stl") && e.contains("seasonal"), "{e}");
    let e = refused(json!(["stl", ["col", "x"], {"period": 4, "window_rows": 7}]));
    assert!(e.contains("fewer than two periods"), "{e}");
    let e = refused(json!(["mstl", ["col", "x"], {"periods": [4, 6], "window_rows": 12}]));
    assert!(
        e.contains("period 6") && e.contains("half the window"),
        "{e}"
    );
    let e = refused(json!(["stl", ["col", "x"], {"period": 24, "window_rows": 40000}]));
    assert!(e.contains("320024 bytes") && e.contains("262144"), "{e}");
    let e = refused(json!(["mstl", ["col", "x"], {"periods": [4, 6], "component": "seasonal"}]));
    assert!(e.contains("needs season"), "{e}");
    let e = refused(json!(["stl", ["col", "x"], {"period": 4, "closed": "both"}]));
    assert!(e.contains("\"left\" or \"right\""), "{e}");
    let e = refused(json!(["stl", ["col", "x"], {"period": 4, "half_life": 3}]));
    assert!(e.contains("not one of its parameters"), "{e}");
}
