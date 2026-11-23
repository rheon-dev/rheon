//! `ewm_moments` through the bank (docs/PLAN.md task 554): the spec's terms
//! read against its features, one output a term in the spec's order and
//! `se_<term>` after them, the core model's numbers to the bit, the same
//! bits at any chunking, a spec that round-trips through JSON and TOML, and
//! the refusals by name.

use polars::prelude::*;
use rheon_core::{
    Decay, EwmMoments, EwmMomentsCfg, MomentKind, MomentLeg, MomentTerm, OnlineModel, Stamp,
};
use rheon_polars::{Bank, Spec, output_fields};

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// One group's rows: a clock in quarters of a unit (exact sums, rows on a
/// lag's and a window's edge) with a gap of 9.5 every 97 rows; `x` a level
/// of 3 plus a skewed draw (a sum of two exponentials), `y` partly `x` a row
/// back; a weight in `[1/4, 5/4)`, 0 on the first row and every 13th.
struct Rows {
    t: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
}

fn rows(n: usize, seed: u64) -> Rows {
    let mut r = lcg(seed);
    let mut out = Rows {
        t: Vec::new(),
        x: Vec::new(),
        y: Vec::new(),
        w: Vec::new(),
    };
    let (mut t, mut prev) = (0.0, 3.0);
    for i in 0..n {
        t += 0.25 * (1.0 + (r() * 6.0).floor());
        if i % 97 == 96 {
            t += 9.5;
        }
        let x = 3.0 - r().ln() - 0.5 * r().ln();
        out.t.push(t);
        out.x.push(x);
        out.y.push(0.6 * prev + 0.8 * (r() - 0.5));
        out.w.push(if i % 13 == 0 { 0.0 } else { 0.25 + r() });
        prev = x;
    }
    out
}

fn frame(groups: &[(&str, usize, u64)]) -> DataFrame {
    let (mut g, mut t, mut x, mut y, mut w) = (vec![], vec![], vec![], vec![], vec![]);
    for &(name, n, seed) in groups {
        let r = rows(n, seed);
        g.extend(std::iter::repeat_n(name.to_string(), n));
        t.extend(r.t);
        x.extend(r.x);
        y.extend(r.y);
        w.extend(r.w);
    }
    df!("g" => g, "t" => t, "x" => x, "y" => y, "w" => w).unwrap()
}

fn spec(json: &str) -> Spec {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("{json}: {e}"))
}

/// The terms of a spec with every kind, a cross term and a lag: written out
/// of alphabetical order, which the outputs keep.
const TERMS: &str = r#""terms": {"m3": {"x": 3}, "kurt": {"x": 4}, "xy2": {"x": 1, "y": 2},
    "ac1": {"x": 1, "x@1": 1}, "rho": {"x": 1, "y@1.5": 1}, "k4": {"x": 4}},
    "kind": {"kurt": "standardized", "rho": "standardized", "k4": "cumulant"}"#;

/// Three specs over the same rows: decayed on the clock with clock lags, a
/// gap cap that breaks the lags, and standard errors; a window of rows with
/// row lags and no clock; a window of the clock with an as-of tolerance.
fn specs() -> Vec<Spec> {
    vec![
        spec(&format!(
            r#"{{"name": "ew", "model": {{"type": "ewm_moments", {TERMS}, "se": true}},
                "features": [], "clock": "t", "gap_cap": 5.0, "group": "g",
                "weight": "w", "half_life": 20.0}}"#
        )),
        spec(
            r#"{"name": "rows", "model": {"type": "ewm_moments", "window_rows": 30,
                "terms": {"m2": {"x": 2}, "ac2": {"x": 1, "x@2": 1}, "sk": {"y": 3}},
                "kind": {"sk": "standardized"}},
                "features": ["x", "y"], "group": "g", "weight": "w", "half_life": "inf"}"#,
        ),
        spec(&format!(
            r#"{{"name": "win", "model": {{"type": "ewm_moments", {TERMS},
                "window_size": 12.0, "lag_tolerance": 0.5}},
                "features": ["x", "y"], "clock": "t", "gap_cap": 1e9, "group": "g",
                "weight": "w", "half_life": 30.0}}"#
        )),
    ]
}

fn run(specs: &[Spec], df: &DataFrame, size: usize) -> DataFrame {
    let mut bank = Bank::new(specs.to_vec()).unwrap();
    let mut out: Option<DataFrame> = None;
    let mut start = 0;
    while start < df.height() {
        let piece = df.slice(start as i64, size);
        let got = DataFrame::new(piece.height(), bank.fit_predict(&piece).unwrap()).unwrap();
        match out.as_mut() {
            None => out = Some(got),
            Some(o) => {
                o.vstack_mut(&got).unwrap();
            }
        }
        start += size;
    }
    out.unwrap()
}

fn field(out: &DataFrame, spec: &str, name: &str) -> Vec<Option<f64>> {
    out.column(spec)
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap_or_else(|e| panic!("{spec}.{name}: {e}"))
        .f64()
        .unwrap()
        .iter()
        .collect()
}

fn bits(v: &[Option<f64>]) -> Vec<Option<u64>> {
    v.iter().map(|x| x.map(f64::to_bits)).collect()
}

/// Each output is named after its term, in the order the spec writes the
/// terms; under `se` each term's standard error follows the terms, in the
/// same order; then the weight and the readiness fields. A field's columns
/// are its legs' columns, lags taken off.
#[test]
fn each_output_is_named_after_its_term_in_the_specs_order() {
    let specs = specs();
    let terms = ["m3", "kurt", "xy2", "ac1", "rho", "k4"];
    let mut want: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
    want.extend(terms.iter().map(|t| format!("se_{t}")));
    want.extend(["weight_sum", "settled_frac", "withheld_reason"].map(String::from));
    assert_eq!(output_fields(&specs[0]), want);
    let mut filled = specs[0].clone();
    filled.check().unwrap();
    assert_eq!(filled.features, ["x", "y"], "the features the terms name");
    let index = rheon_polars::output_index(&filled);
    let rho = index.iter().find(|f| f.field == "se_rho").unwrap();
    assert_eq!(rho.kind, "se_moment");
    assert_eq!(rho.columns.as_deref(), Some(&["x".into(), "y".into()][..]));
    assert_eq!(
        output_fields(&specs[1]),
        [
            "m2",
            "ac2",
            "sk",
            "weight_sum",
            "settled_frac",
            "withheld_reason"
        ]
    );
    // A grid of half-lives: each instance's terms and weight, suffixed.
    let grid = spec(
        r#"{"name": "g", "model": {"type": "ewm_moments", "terms": {"v": {"x": 2}}},
            "features": ["x"], "half_life": [10.0, 50.0]}"#,
    );
    assert_eq!(
        output_fields(&grid),
        [
            "v@h10",
            "weight_sum@h10",
            "settled_frac@h10",
            "withheld_reason@h10",
            "v@h50",
            "weight_sum@h50",
            "settled_frac@h50",
            "withheld_reason@h50"
        ]
    );
    // The bank writes the same fields, and every term reports on this
    // stream once it has its rows.
    let df = frame(&[("a", 300, 11)]);
    let out = run(&specs, &df, 300);
    for s in &specs {
        let names: Vec<String> = out
            .column(&s.name)
            .unwrap()
            .struct_()
            .unwrap()
            .fields_as_series()
            .iter()
            .map(|f| f.name().to_string())
            .collect();
        assert_eq!(names, output_fields(s), "{}", s.name);
    }
    for t in terms {
        let v = field(&out, "ew", t);
        assert!(
            v.iter().filter(|v| v.is_some()).count() > 250,
            "{t}: {:?}",
            &v[..10]
        );
    }
}

/// Fed as one chunk, in chunks of 41 and of 7, every output of every spec
/// is the same, to the bit: two groups, breaks at the gaps on one spec,
/// zero-weight rows (the first among them), each window.
#[test]
fn chunks_change_no_bit() {
    let df = frame(&[("a", 300, 5), ("b", 260, 6)]);
    let specs = specs();
    let whole = run(&specs, &df, df.height());
    for size in [41, 7] {
        let cut = run(&specs, &df, size);
        for s in &specs {
            for name in output_fields(s) {
                let a = whole.column(&s.name).unwrap().struct_().unwrap();
                let b = cut.column(&s.name).unwrap().struct_().unwrap();
                let (fa, fb) = (
                    a.field_by_name(&name).unwrap(),
                    b.field_by_name(&name).unwrap(),
                );
                if fa.dtype() == &DataType::Float64 {
                    let (va, vb): (Vec<_>, Vec<_>) = (
                        fa.f64().unwrap().iter().collect(),
                        fb.f64().unwrap().iter().collect(),
                    );
                    assert_eq!(bits(&va), bits(&vb), "{}.{name} at {size}", s.name);
                } else {
                    assert!(fa.equals_missing(&fb), "{}.{name} at {size}", s.name);
                }
            }
        }
    }
}

/// The core model's terms, by hand: the spec's terms with `x` column 0 and
/// `y` column 1, each kind and each statistic's default bias.
fn core_terms() -> Vec<MomentTerm> {
    let term = |name: &str, powers: Vec<(MomentLeg, u32)>, kind: MomentKind| {
        let order = powers.iter().map(|&(_, p)| p).sum();
        MomentTerm {
            name: name.into(),
            powers,
            kind,
            bias: MomentTerm::default_bias(kind, order),
        }
    };
    let (x, y) = (MomentLeg::of(0), MomentLeg::of(1));
    vec![
        term("m3", vec![(x, 3)], MomentKind::Central),
        term("kurt", vec![(x, 4)], MomentKind::Standardized),
        term("xy2", vec![(x, 1), (y, 2)], MomentKind::Central),
        term(
            "ac1",
            vec![(x, 1), (MomentLeg::lagged(0, 1.0), 1)],
            MomentKind::Central,
        ),
        term(
            "rho",
            vec![(x, 1), (MomentLeg::lagged(1, 1.5), 1)],
            MomentKind::Standardized,
        ),
        term("k4", vec![(x, 4)], MomentKind::Cumulant),
    ]
}

/// The bank's outputs are the core model's on the same rows, to the bit:
/// decayed with clock lags and standard errors, and over a window of the
/// clock with an as-of tolerance; `weight_sum` is the model's weight. The
/// bank hands each step its row's stamp, and reads a row of weight 0 at its
/// own (`predict_at`): read at the clock summed from the steps, which
/// begins at the first row that counts, a window's edge there was off by
/// the clock's origin.
#[test]
fn the_banks_terms_are_the_models() {
    let df = frame(&[("a", 400, 9)]);
    let r = rows(400, 9);
    type Case = (&'static str, &'static str, fn(&mut EwmMomentsCfg));
    let cases: [Case; 2] = [
        (
            r#""se": true"#,
            r#""half_life": 20.0"#,
            |c: &mut EwmMomentsCfg| {
                c.decay = Decay::Halflife(20.0);
                c.se = true;
            },
        ),
        (
            r#""window_size": 12.0, "lag_tolerance": 0.5"#,
            r#""half_life": 30.0"#,
            |c: &mut EwmMomentsCfg| {
                c.decay = Decay::Halflife(30.0);
                c.window_clock = Some(12.0);
                c.lag_tolerance = Some(0.5);
            },
        ),
    ];
    for (model, decay, set) in cases {
        let sp = spec(&format!(
            r#"{{"name": "m", "model": {{"type": "ewm_moments", {TERMS}, {model}}},
                "features": ["x", "y"], "clock": "t", "gap_cap": 1e9, "weight": "w",
                {decay}}}"#
        ));
        let out = run(std::slice::from_ref(&sp), &df, 400);
        let mut cfg = EwmMomentsCfg::new(Decay::Halflife(1.0), core_terms());
        cfg.clocked = true;
        set(&mut cfg);
        let mut m = EwmMoments::new(cfg).unwrap();
        let n_out = m.n_outputs();
        let mut want: Vec<Vec<Option<f64>>> = vec![Vec::new(); n_out + 1];
        // A row of weight 0 is no row of the clock: it is read from the
        // state over its step from the last row that counted, and the next
        // row steps from there too.
        let mut last: Option<f64> = None;
        for i in 0..r.t.len() {
            let d = last.map_or(0.0, |l| r.t[i] - l);
            let (x, stamp) = ([r.x[i], r.y[i]], Stamp::from(r.t[i]));
            let step = if r.w[i] > 0.0 {
                last = Some(r.t[i]);
                m.stamp_next(stamp);
                m.step(&x, &[], d, r.w[i])
            } else {
                m.predict_at(&x, d, stamp)
            };
            for (j, v) in step.pred.iter().enumerate() {
                want[j].push(Some(*v).filter(|v| v.is_finite()));
            }
            want[n_out].push(Some(step.n_eff));
        }
        let names = output_fields(&sp);
        for (j, w) in want.iter().enumerate() {
            let name = if j == n_out {
                "weight_sum"
            } else {
                names[j].as_str()
            };
            let got = field(&out, "m", name);
            let first = (0..w.len()).find(|&i| bits(&got[i..=i]) != bits(&w[i..=i]));
            assert_eq!(
                first,
                None,
                "{model}: {name} at row {first:?}: {:?} against {:?}",
                first.map(|i| (got[i], r.t[i], r.w[i])),
                first.map(|i| w[i])
            );
        }
    }
}

/// The spec writes and reads back as it was, through JSON and TOML, the
/// terms and the kinds in the order written.
#[test]
fn the_spec_round_trips() {
    for s in specs() {
        let json = serde_json::to_string(&s).unwrap();
        let back: Spec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s, "{json}");
        let toml = toml::to_string(&s).unwrap();
        let back: Spec = toml::from_str(&toml).unwrap();
        assert_eq!(back, s, "{toml}");
    }
    let s = &specs()[0];
    let json = serde_json::to_string(&s.model).unwrap();
    let at = |k: &str| json.find(k).unwrap_or_else(|| panic!("{k} in {json}"));
    assert!(at("\"m3\"") < at("\"kurt\"") && at("\"kurt\"") < at("\"xy2\""));
    assert!(at("\"ac1\"") < at("\"rho\"") && at("\"rho\"") < at("\"k4\""));
}

fn check(json: &str) -> Result<(), String> {
    let mut s: Spec = serde_json::from_str(json).map_err(|e| format!("did not parse: {e}"))?;
    s.check()
}

fn moments(model: &str, extra: &str) -> String {
    format!(
        r#"{{"name": "m", "model": {{"type": "ewm_moments", {model}}}, "features": ["x", "y"],
            "half_life": 10.0{extra}}}"#
    )
}

fn refused(json: &str, want: &[&str]) {
    let e = check(json).expect_err(&format!("accepted: {json}"));
    for w in want {
        assert!(e.contains(w), "wanted {w:?} in: {e}");
    }
    assert!(e.starts_with("spec \"m\": "), "the spec is not named: {e}");
}

fn accepted(json: &str) {
    check(json).unwrap_or_else(|e| panic!("refused: {e}\n{json}"));
}

/// What the terms, the kinds, the bias, the lags and the window cannot be
/// is refused, naming the term or the parameter and the value.
#[test]
fn a_spec_it_cannot_read_is_refused_by_name() {
    let clock = r#", "clock": "t", "gap_cap": 10.0"#;
    accepted(&moments(r#""terms": {"m2": {"x": 2}}"#, ""));
    accepted(&moments(r#""terms": {"ac": {"x": 1, "x@3": 1}}"#, ""));
    accepted(&moments(
        r#""terms": {"ac": {"x": 1, "x@0.5": 1}}, "lag_tolerance": 0.25"#,
        clock,
    ));
    refused(
        &moments(
            r#""terms": {"k11": {"x": 1, "y": 1}}, "kind": "cumulant""#,
            "",
        ),
        &["\"k11\"", "joint cumulant"],
    );
    refused(
        &moments(r#""terms": {"m5": {"x": 5}}, "bias": false"#, ""),
        &["\"m5\"", "bias=False at order 5"],
    );
    accepted(&moments(r#""terms": {"m4": {"x": 4}}, "bias": false"#, ""));
    refused(
        &moments(r#""terms": {"m2": {"z": 2}}"#, ""),
        &["\"m2\"", "names the column \"z\"", "not among the features"],
    );
    refused(
        &moments(r#""terms": {"ac": {"x": 1, "z@1": 1}}"#, ""),
        &["\"ac\"", "names the column \"z\""],
    );
    refused(
        &moments(r#""terms": {"ac": {"x": 1, "x@soon": 1}}"#, ""),
        &["\"ac\"", "neither a number nor a duration"],
    );
    // Without a clock a lag is learned rows: a whole number, no duration
    // and no tolerance.
    refused(
        &moments(r#""terms": {"ac": {"x": 1, "x@1.5": 1}}"#, ""),
        &["\"ac\"", "1.5 rows, which is not a whole number"],
    );
    refused(
        r#"{"name": "m", "model": {"type": "ewm_moments", "terms": {"ac": {"x": 1, "x@5m": 1}}},
            "features": ["x"], "half_life": "inf"}"#,
        &["terms is a duration, but the spec has no clock"],
    );
    refused(
        &moments(
            r#""terms": {"ac": {"x": 1, "x@1": 1}}, "lag_tolerance": 1.0"#,
            "",
        ),
        &["lag_tolerance is the as-of tolerance of a lag on a clock"],
    );
    refused(
        &moments(r#""terms": {"m2": {"x": 2}}, "window_size": 20.0"#, ""),
        &["window_size = 20 is a span of the clock, but the spec has no clock"],
    );
    refused(
        &moments(
            r#""terms": {"m2": {"x": 2}}, "window_size": 20.0, "window_rows": 20"#,
            clock,
        ),
        &["window_rows (a count of rows) and window_size"],
    );
    refused(
        &moments(r#""terms": {"m2": {"x": 2}}, "window_rows": 0"#, ""),
        &["window_rows must be at least 1"],
    );
    refused(
        &moments(r#""terms": {"m2": {"x": 2}}, "kind": "skew""#, ""),
        &["kind must be", "\"skew\""],
    );
    refused(
        &moments(r#""terms": {"m2": {"x": 2}}, "kind": {"m2": "excess"}"#, ""),
        &["\"m2\"", "\"excess\""],
    );
    refused(
        &moments(r#""terms": {"m2": {"x": 2}}, "kind": {"m3": "raw"}"#, ""),
        &["kind names the term \"m3\", which terms does not have"],
    );
    refused(
        &moments(r#""terms": {"s1": {"x": 1}}, "kind": "standardized""#, ""),
        &["\"s1\"", "standardized at order 1"],
    );
    refused(
        &moments(r#""terms": {"m0": {"x": 0}}"#, ""),
        &["\"m0\"", "order 0"],
    );
    refused(&moments(r#""terms": {}"#, ""), &["terms names no term"]);
    refused(
        &moments(r#""terms": {"m4": {"x": 4}}, "window_rows": 100000"#, ""),
        &["over max_state_bytes (262144)"],
    );
    refused(
        &moments(r#""terms": {"m2": {"x": 2}}, "max_state_bytes": 0"#, ""),
        &["max_state_bytes must be above 0"],
    );
    // A term named as another output names two fields alike.
    refused(
        &moments(r#""terms": {"weight_sum": {"x": 2}}"#, ""),
        &["weight_sum"],
    );
    // No `terms` at all is the parser's to name.
    let e = check(&moments(r#""se": true"#, "")).unwrap_err();
    assert!(e.contains("missing field `terms`"), "{e}");
}
