//! The `invert` node: the interpreter against direct arithmetic, each
//! reason a row is withheld for, the refusals when the node is read, and the
//! node as a Polars expression, chunked one way and another.

use serde_json::json;

use super::*;

fn node(f: Value, equals: Value, lo: Value, hi: Value, params: Value) -> Invert {
    Invert::from_json(&json!(["invert", f, equals, lo, hi, params])).expect("a valid node")
}

fn x() -> Value {
    json!(["unknown"])
}

fn c(name: &str) -> Value {
    json!(["col", name])
}

fn l(v: f64) -> Value {
    json!(["lit", v])
}

/// `x³ + a·x + exp(b·x)`, increasing for `a, b >= 0`, equal to the column
/// `t`, on `[−5, 5]`.
fn monotone(params: Value) -> Invert {
    let f = json!([
        "+",
        ["+", ["**", x(), l(3.0)], ["*", c("a"), x()]],
        ["exp", ["*", c("b"), x()]]
    ]);
    node(f, c("t"), l(-5.0), l(5.0), params)
}

fn solve(inv: &Invert, row: &[f64]) -> RootReport {
    inv.solve_row(row, &mut Vec::new())
}

#[test]
fn the_interpreter_gives_direct_arithmetic_on_every_node() {
    let cols = ["p", "q"];
    let row = [1.7, -0.3];
    let (p, q, xv) = (row[0], row[1], 0.45);
    let cases: Vec<(Value, f64)> = vec![
        (json!(["+", c("p"), x()]), p + xv),
        (json!(["-", c("p"), x()]), p - xv),
        (json!(["*", c("q"), x()]), q * xv),
        (json!(["/", c("q"), x()]), q / xv),
        (json!(["**", c("p"), x()]), p.powf(xv)),
        (json!(["log", c("p"), x()]), p.log(xv)),
        (json!(["neg", x()]), -xv),
        (json!(["abs", c("q")]), q.abs()),
        (json!(["exp", x()]), xv.exp()),
        (json!(["sqrt", c("p")]), p.sqrt()),
        (json!(["cast", c("q"), "Float64"]), q),
        (json!(["lit", 3]), 3.0),
    ];
    for (tree, want) in cases {
        let mut names = vec![cols[0].to_string(), cols[1].to_string()];
        let prog = Program::compile(&tree, &mut names, "function").unwrap();
        assert_eq!(names, cols, "{tree}");
        let got = prog.eval(&row, xv, &mut Vec::new());
        assert_eq!(got.value.to_bits(), want.to_bits(), "{tree}");
        assert!(
            got.error >= 0.0 && got.error <= 1e-14 * (1.0 + want.abs()),
            "{tree}: {got:?}"
        );
    }
}

#[test]
fn the_running_bound_covers_a_cancellation() {
    // (1 + 1e-15·x) − 1 at x = 1 loses most of its digits: its bound must
    // cover the true value, 1e-15, which is 1.1102230246251565e-15 off.
    let tree = json!(["-", ["+", l(1.0), ["*", l(1e-15), x()]], l(1.0)]);
    let prog = Program::compile(&tree, &mut Vec::new(), "function").unwrap();
    let got = prog.eval(&[], 1.0, &mut Vec::new());
    assert!((got.value - 1e-15).abs() <= got.error, "{got:?}");
    // An operand's bound reaches through a division by a value near it.
    let near = binary(Op::Div, Bounded::exact(1.0), Bounded::new(1e-20, 1e-20));
    assert_eq!(near.error, f64::INFINITY);
    let exp = unary(Op::Exp, Bounded::new(2.0, 1e-10));
    assert!(exp.error >= 2.0f64.exp() * 1e-10);
    let ln = binary(
        Op::Log,
        Bounded::new(1.0, 2.0),
        Bounded::exact(std::f64::consts::E),
    );
    assert_eq!(ln.error, f64::INFINITY);
    let pow = binary(Op::Pow, Bounded::new(-2.0, 0.0), Bounded::new(3.0, 1e-12));
    assert_eq!(pow.error, f64::INFINITY);
    let sqrt = unary(Op::Sqrt, Bounded::new(4.0, 0.5));
    assert!(sqrt.error >= 4.0f64.sqrt() - 3.5f64.sqrt());
}

#[test]
fn a_root_is_found_to_the_tolerance() {
    let inv = monotone(json!({"increasing": true}));
    assert_eq!(inv.columns(), ["a", "b", "t"]);
    let (a, b) = (1.3, 0.4);
    let f = |z: f64| z.powf(3.0) + a * z + (b * z).exp();
    for root in [-4.2, -0.5, 0.0, 1e-9, 2.75] {
        let r = solve(&inv, &[a, b, f(root)]);
        let got = r.root.expect("a certified root");
        assert!((got - root).abs() <= 1e-11, "{root}: {got}");
        assert!(r.lo <= got && got <= r.hi && r.evaluations <= r.bound);
    }
}

#[test]
fn a_decreasing_function_gives_the_bits_of_its_negation() {
    let up = monotone(json!({"increasing": true}));
    let f = json!([
        "neg",
        [
            "+",
            ["+", ["**", x(), l(3.0)], ["*", c("a"), x()]],
            ["exp", ["*", c("b"), x()]]
        ]
    ]);
    let down = node(
        f,
        json!(["neg", c("t")]),
        l(-5.0),
        l(5.0),
        json!({"increasing": false}),
    );
    for t in [-60.0, -1.0, 0.3, 7.0, 90.0] {
        let row = [0.8, 0.2, t];
        assert_eq!(solve(&up, &row), solve(&down, &row), "{t}");
    }
}

#[test]
fn each_reason_is_given_where_it_applies() {
    let inv = monotone(json!({"increasing": true}));
    let why = |r: RootReport| r.root.err();
    let ok = [1.0, 0.5, 2.0];
    assert_eq!(why(solve(&inv, &ok)), None);
    for bad in [f64::NAN, f64::INFINITY, 2e100] {
        assert_eq!(
            why(solve(&inv, &[1.0, bad, 2.0])),
            Some(Withheld::InputMissing)
        );
    }
    assert_eq!(
        why(solve(&inv, &[1.0, 0.5, -1e6])),
        Some(Withheld::BelowRange)
    );
    assert_eq!(
        why(solve(&inv, &[1.0, 0.5, 1e6])),
        Some(Withheld::AboveRange)
    );
    let wrong = monotone(json!({"increasing": false}));
    assert_eq!(why(solve(&wrong, &ok)), Some(Withheld::WrongDirection));
    // lo >= hi, from the row's own columns.
    let rev = node(
        json!(["-", x(), c("t")]),
        l(0.0),
        c("lo"),
        c("hi"),
        json!({"increasing": true}),
    );
    assert_eq!(
        why(solve(&rev, &[1.0, 3.0, 2.0])),
        Some(Withheld::BracketInvalid)
    );
    assert_eq!(
        why(solve(&rev, &[1.0, 3.0, f64::NAN])),
        Some(Withheld::InputMissing)
    );
    // sqrt of a negative at the lower end: a domain error.
    let dom = node(
        json!(["sqrt", x()]),
        c("t"),
        l(-1.0),
        l(4.0),
        json!({"increasing": true}),
    );
    assert_eq!(why(solve(&dom, &[1.0])), Some(Withheld::NotFinite));
    // `(1 + 1e-16·x) − 1` is certain in sign at ±10, but near its root its
    // value is below the rounding of `1 + ...`: not determined.
    let flat = node(
        json!(["-", ["+", l(1.0), ["*", l(1e-16), x()]], l(1.0)]),
        c("t"),
        l(-10.0),
        l(10.0),
        json!({"increasing": true}),
    );
    let r = solve(&flat, &[0.0]);
    assert_eq!(why(r), Some(Withheld::NotDetermined), "{r:?}");
    // The same slope with no cancellation is certified.
    let steep = node(
        json!(["*", l(1e-16), x()]),
        c("t"),
        l(-10.0),
        l(10.0),
        json!({"increasing": true}),
    );
    assert_eq!(why(solve(&steep, &[0.0])), None);
}

#[test]
fn a_target_with_its_own_rounding_widens_the_bound() {
    // The target 0.1 + 0.2 is computed: its rounding is part of the
    // function's bound, so a root still comes out where the slope is 1.
    let inv = node(
        x(),
        json!(["+", ["*", l(0.1), c("s")], l(0.2)]),
        l(-1.0),
        l(1.0),
        json!({"increasing": true}),
    );
    let got = solve(&inv, &[1.0]).root.unwrap();
    assert!((got - 0.3).abs() <= 1e-11);
}

#[test]
fn a_malformed_node_is_refused_by_name() {
    let refuse = |v: Value, needle: &str| {
        let e = Invert::from_json(&v).unwrap_err();
        assert!(e.contains(needle), "{e}");
        assert!(e.starts_with("invert: "), "{e}");
    };
    let p = json!({"increasing": true});
    refuse(json!(["col", "x"]), "headed");
    refuse(json!(["invert", x(), c("t"), l(0.0)]), "5 arguments");
    refuse(
        json!(["invert", c("t"), c("t"), l(0.0), l(1.0), p]),
        "does not hold the unknown",
    );
    refuse(
        json!(["invert", x(), x(), l(0.0), l(1.0), p]),
        "target (equals) holds the unknown",
    );
    refuse(
        json!(["invert", x(), l(0.0), x(), l(1.0), p]),
        "lower end holds the unknown",
    );
    refuse(
        json!(["invert", x(), l(0.0), l(0.0), l(1.0), p]),
        "reads no column",
    );
    refuse(
        json!(["invert", ["shift", x()], c("t"), l(0.0), l(1.0), p]),
        "\"shift\"",
    );
    refuse(
        json!(["invert", ["+", x()], c("t"), l(0.0), l(1.0), p]),
        "takes 2 arguments",
    );
    refuse(
        json!(["invert", x(), ["lit", "a"], l(0.0), l(1.0), p]),
        "finite number",
    );
    refuse(
        json!(["invert", x(), ["cast", c("t"), "Int64"], l(0.0), l(1.0), p]),
        "Float64",
    );
    refuse(
        json!(["invert", x(), c("t"), l(0.0), l(1.0), {}]),
        "increasing is not set",
    );
    refuse(
        json!(["invert", x(), c("t"), l(0.0), l(1.0), {"increasing": 1}]),
        "true or false",
    );
    refuse(
        json!(["invert", x(), c("t"), l(0.0), l(1.0), {"increasing": true, "xtol": 0.0}]),
        "xtol",
    );
    refuse(
        json!(["invert", x(), c("t"), l(0.0), l(1.0), {"increasing": true, "rtol": 1e-17}]),
        "rtol",
    );
    refuse(
        json!(["invert", x(), c("t"), l(0.0), l(1.0), {"increasing": true, "xtol": "a"}]),
        "a number",
    );
    refuse(
        json!(["invert", x(), c("t"), l(0.0), l(1.0), {"increasing": true, "tol": 1}]),
        "\"tol\"",
    );
    refuse(
        json!(["invert", x(), c("t"), l(0.0), l(1.0), [1]]),
        "an object",
    );
    assert!(Invert::parse("[").unwrap_err().contains("not JSON"));
}

fn frame(n: usize) -> DataFrame {
    let a: Vec<f64> = (0..n).map(|i| 0.1 + (i % 7) as f64 * 0.3).collect();
    let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 * 0.2).collect();
    let t: Vec<Option<f64>> = (0..n)
        .map(|i| (i % 11 != 3).then_some(-200.0 + i as f64 * 2.2))
        .collect();
    df!("a" => a, "b" => b, "t" => t).unwrap()
}

fn run(inv: &Invert, df: DataFrame) -> DataFrame {
    df.lazy()
        .select([inv.to_expr().alias("iv")])
        .collect()
        .unwrap()
}

#[test]
fn the_expression_gives_each_rows_solve() {
    let inv = monotone(json!({"increasing": true}));
    let df = frame(200);
    let out = run(&inv, df.clone());
    let got = out.column("iv").unwrap().f64().unwrap();
    let cols: Vec<Vec<Option<f64>>> = ["a", "b", "t"]
        .iter()
        .map(|c| df.column(c).unwrap().f64().unwrap().iter().collect())
        .collect();
    let mut kinds = std::collections::HashSet::new();
    for i in 0..df.height() {
        let row: Vec<f64> = cols.iter().map(|c| c[i].unwrap_or(f64::NAN)).collect();
        let want = solve(&inv, &row);
        kinds.insert(want.root.err());
        assert_eq!(
            got.get(i).map(f64::to_bits),
            want.root.ok().map(f64::to_bits),
            "row {i}"
        );
    }
    // Certified rows, a null target and targets past either end.
    for k in [
        None,
        Some(Withheld::InputMissing),
        Some(Withheld::BelowRange),
        Some(Withheld::AboveRange),
    ] {
        assert!(kinds.contains(&k), "{k:?} in {kinds:?}");
    }
}

#[test]
fn the_detail_struct_carries_the_bracket_the_count_and_the_reason() {
    let inv = monotone(json!({"increasing": true, "detail": true}));
    assert!(inv.detail());
    let df = frame(50);
    let out = run(&inv, df.clone()).column("iv").unwrap().clone();
    assert_eq!(out.dtype(), &inv.dtype());
    let s = out.struct_().unwrap();
    let field = |n: &str| s.field_by_name(n).unwrap();
    let root = field("root");
    let reason = field("withheld_reason");
    let (evals, bound) = (field("evaluations"), field("bound"));
    let (lo, hi) = (field("lo"), field("hi"));
    for i in 0..df.height() {
        let r = root.f64().unwrap().get(i);
        let why = reason.str().unwrap().get(i);
        assert_eq!(r.is_some(), why.is_none(), "row {i}");
        assert!(evals.u32().unwrap().get(i) <= bound.u32().unwrap().get(i));
        if let Some(r) = r {
            assert!(
                lo.f64().unwrap().get(i).unwrap() <= r && r <= hi.f64().unwrap().get(i).unwrap()
            );
        }
        if why == Some("input_missing") {
            assert_eq!(lo.f64().unwrap().get(i), None);
        }
    }
    let names: Vec<String> = s
        .fields_as_series()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    assert_eq!(names, INVERT_DETAIL_FIELDS);
}

#[test]
fn one_chunk_or_many_give_the_same_bits() {
    let inv = monotone(json!({"increasing": true, "detail": true}));
    let whole = frame(300);
    let one = run(&inv, whole.clone());
    for size in [1usize, 7, 64] {
        let mut parts = whole.slice(0, size);
        let mut at = size;
        while at < whole.height() {
            parts.vstack_mut(&whole.slice(at as i64, size)).unwrap();
            at += size;
        }
        assert!(parts.first_col_n_chunks() > 1 || size >= 300);
        let many = run(&inv, parts);
        assert!(one.equals_missing(&many), "chunks of {size}");
    }
}

#[test]
fn a_constant_column_is_broadcast_and_a_cast_is_read() {
    // An integer column is cast to Float64; the bracket reads a column.
    let inv = node(
        json!(["-", ["*", c("k"), x()], c("t")]),
        l(0.0),
        l(-10.0),
        c("hi"),
        json!({"increasing": true}),
    );
    let df = df!("k" => [2i64, 4, 5], "t" => [1.0, 2.0, 3.0], "hi" => [10.0, 10.0, 10.0]).unwrap();
    let out = run(&inv, df);
    let got: Vec<f64> = out
        .column("iv")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    for (g, w) in got.iter().zip([0.5, 0.5, 0.6]) {
        assert!((g - w).abs() <= 1e-11, "{g} {w}");
    }
    let one = inv
        .apply(
            "v".into(),
            &[
                Column::new("k".into(), [2.0]),
                Column::new("t".into(), [1.0, 3.0]),
                Column::new("hi".into(), [10.0]),
            ],
        )
        .unwrap();
    assert_eq!(one.len(), 2);
    let bad = inv.apply(
        "v".into(),
        &[
            Column::new("k".into(), [2.0, 1.0]),
            Column::new("t".into(), [1.0, 3.0, 4.0]),
            Column::new("hi".into(), [10.0]),
        ],
    );
    assert!(bad.is_err());
}
