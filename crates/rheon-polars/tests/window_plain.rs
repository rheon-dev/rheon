//! The window operators' plain-run loop is `Windows::push` (docs/PLAN.md task
//! 429). A run of distinct, increasing stamps in one group, within the cap and
//! its session, with a value in every input, is folded a block at a time by a
//! loop of its own; any other row is the general `push`'s, and the loop goes on
//! after it. The loop takes a datetime, a float and an integer clock and a
//! row-count clock (no clock column), and a row that starts its group's windows
//! over: a gap past the cap at one key or many, the first row after a cut
//! (task 442). Here every stream below runs twice, once with the plain loop and
//! once with it switched off, as four chunkings (two of them saved and loaded
//! in the middle), and every output bit, every null and every byte of the saved
//! state must be equal. The streams are plain for long stretches, broken by the
//! rows that are not -- a null in an input, a tied stamp, a gap past the cap, a
//! session change, a step back that restarts -- so a block ends and hands over
//! mid-chunk in each way; and the loop must have taken rows (`plain_rows`), or
//! a green run would say nothing about it.
use polars::prelude::*;
use rheon_polars::{WindowsConfig, WindowsRun};
use serde_json::{Value, json};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// How a stream breaks its plain stretches, each a probability per row.
#[derive(Clone, Copy, Default, Debug)]
struct Breaks {
    null: f64,
    tie: f64,
    cap: f64,
    session: f64,
    back: f64,
    /// A key of its own that a row has now and then: it falls silent past the
    /// cap between its rows, which cuts its windows while the stream goes on.
    rare: f64,
    /// A step of the cap, or a unit either side of it: the edge of "past the
    /// cap", where an exact test and a rounded one part.
    edge: f64,
    /// A NaN, +inf or -inf in place of `x`'s value, never a null: a column
    /// with no null is read where it lies (review 459, C7).
    nonfinite: f64,
    /// Where the integer clock starts, in ms (0: the usual 1.7e12): near
    /// either end of an i64, the steps are differences no i64 subtraction
    /// of the extremes holds (review 459, C7).
    int_base: i64,
}

/// The clock a case runs on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Clk {
    Datetime,
    Float,
    /// An integer column, in milliseconds.
    Int,
    /// No clock column: each row steps by one.
    Rows,
}

impl Clk {
    /// A time of `secs` seconds in this clock's units, as an operator's
    /// parameter: a duration string, or a number (a row count on `Rows`).
    fn t(self, secs: f64) -> Value {
        match self {
            Clk::Datetime => json!(format!("{secs}s")),
            Clk::Float | Clk::Rows => json!(secs),
            Clk::Int => json!(secs * 1000.0),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Clk::Datetime => "datetime",
            Clk::Float => "float",
            Clk::Int => "integer",
            Clk::Rows => "row-count",
        }
    }
}

struct Data {
    ts_ns: Vec<i64>,
    ts_f: Vec<f64>,
    ts_i: Vec<i64>,
    x: Vec<Option<f64>>,
    y: Vec<Option<f64>>,
    g: Vec<i64>,
    s: Vec<i64>,
}

fn data(n: usize, seed: u64, b: Breaks, keys: u32) -> Data {
    let mut r = Rng(seed);
    let (mut t, mut m, mut sess) = (0.0f64, 100.0f64, 0i64);
    let mut d = Data {
        ts_ns: vec![],
        ts_f: vec![],
        ts_i: vec![],
        x: vec![],
        y: vec![],
        g: vec![],
        s: vec![],
    };
    for _ in 0..n {
        let u = r.next();
        let step = if r.next() < b.tie {
            0.0
        } else if r.next() < b.cap {
            400.0
        } else if r.next() < b.edge {
            [29.999, 30.0, 30.001][(r.next() * 3.0) as usize % 3]
        } else {
            -0.5 * u.ln()
        };
        if r.next() < b.back {
            t -= 3.0;
        } else {
            t += step;
        }
        if r.next() < b.session {
            sess += 1;
        }
        m += (r.next() - 0.5) * 0.02;
        d.ts_ns.push((t * 1e9) as i64 + 1_700_000_000_000_000_000);
        d.ts_f.push(((t * 1e9) as i64) as f64 / 1e9);
        let base = if b.int_base == 0 {
            1_700_000_000_000
        } else {
            b.int_base
        };
        d.ts_i.push((t * 1e3) as i64 + base);
        let nonfinite = r.next() < b.nonfinite;
        d.x.push((r.next() >= b.null).then_some(if nonfinite {
            [f64::NAN, f64::INFINITY, f64::NEG_INFINITY][(r.next() * 3.0) as usize % 3]
        } else {
            m
        }));
        d.y.push((r.next() >= b.null).then_some(m * 0.5 + r.next()));
        d.g.push(if r.next() < b.rare {
            i64::from(keys)
        } else {
            (r.next() * f64::from(keys)) as i64
        });
        d.s.push(sess);
    }
    d
}

fn frame(d: &Data, clk: Clk, grouped: bool, session: bool, a: usize, z: usize) -> DataFrame {
    let mut cols = vec![];
    match clk {
        Clk::Float => cols.push(Column::new("ts".into(), &d.ts_f[a..z])),
        Clk::Datetime => cols.push(
            Column::new("ts".into(), &d.ts_ns[a..z])
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                .unwrap(),
        ),
        Clk::Int => cols.push(Column::new("ts".into(), &d.ts_i[a..z])),
        Clk::Rows => {}
    }
    cols.push(Column::new("x".into(), &d.x[a..z]));
    cols.push(Column::new("y".into(), &d.y[a..z]));
    if grouped {
        cols.push(Column::new("g".into(), &d.g[a..z]));
    }
    if session {
        cols.push(Column::new("s".into(), &d.s[a..z]));
    }
    DataFrame::new_infer_height(cols).unwrap()
}

/// Every output bit and which rows are null, then the state's bytes.
struct H(u64, u64);
impl H {
    fn u(&mut self, v: u64) {
        self.0 = (self.0 ^ v).wrapping_mul(0x100000001b3);
    }
    fn frame(&mut self, f: &DataFrame) {
        for c in f.columns() {
            self.u(c.len() as u64);
            if let Ok(ca) = c.f64() {
                for v in ca.iter() {
                    self.u(v.map_or(0x7ff8_dead_beef, f64::to_bits));
                }
            } else if let Ok(ca) = c.i64() {
                for v in ca.iter() {
                    self.u(v.map_or(1, |x| x as u64));
                }
            }
        }
    }
    fn bytes(&mut self, b: &[u8]) {
        for x in b {
            self.u(u64::from(*x));
        }
        self.1 += b.len() as u64;
    }
}

fn op(name: &str, input: &str, hl: &Value, extra: Value) -> Value {
    let mut o = json!({"half_life": hl});
    if let Value::Object(e) = extra {
        for (k, v) in e {
            o[k] = v;
        }
    }
    json!([name, ["col", input], o])
}

/// The operators of a case, as formulas: every statistic a plain kernel runs,
/// under the four `closed` values, two inputs, `min_samples`, and a kernel of
/// its own for each half-life.
fn formulas(clk: Clk, flavor: usize) -> Vec<Value> {
    let (h5, h2) = (clk.t(5.0), clk.t(2.0));
    let closed = |c: &str| json!({"closed": c});
    let mut f = vec![];
    let mut add = |name: &str, t: Value| f.push(json!({"name": name, "tree": t}));
    match flavor {
        // One mean: the plain case.
        0 => add("mean", op("ewm_mean", "x", &h5, json!({}))),
        // Every statistic of one kernel, and a kernel of another half-life.
        1 => {
            add("mean", op("ewm_mean", "x", &h5, json!({})));
            add("std", op("ewm_std", "x", &h5, json!({})));
            add("var", op("ewm_var", "x", &h5, json!({"bias": true})));
            add("sum", op("ewm_sum", "x", &h5, json!({})));
            add("rate", op("ewm_rate", "x", &h5, json!({})));
            add("mean2", op("ewm_mean", "y", &h2, json!({})));
            add("sum2", op("ewm_sum", "y", &h2, json!({"min_samples": 3})));
        }
        // The other closed values, each its own kernel, two inputs.
        2 => {
            for c in ["left", "both", "none", "right"] {
                add(&format!("mean_{c}"), op("ewm_mean", "x", &h5, closed(c)));
                add(&format!("rate_{c}"), op("ewm_rate", "y", &h5, closed(c)));
                add(&format!("var_{c}"), op("ewm_var", "x", &h2, closed(c)));
            }
        }
        // No decay: a sum and a rate that count the whole stretch.
        3 => {
            add("sum", op("ewm_sum", "x", &json!("inf"), json!({})));
            add(
                "rate",
                op("ewm_rate", "x", &json!("inf"), json!({"closed": "left"})),
            );
            add("mean", op("ewm_mean", "y", &h5, json!({"min_samples": 2})));
        }
        // Windows: the queue's reads and pushes, under every `closed` and
        // `partial`, beside a running kernel.
        4 | 5 => {
            let w = |secs: f64| clk.t(secs);
            add("mean", op("ewm_mean", "x", &h5, json!({})));
            add(
                "windowed",
                op("ewm_mean", "x", &h5, json!({"window_size": w(30.0)})),
            );
            if flavor == 5 {
                for c in ["right", "left", "both", "none"] {
                    let ex = |partial: &str| json!({"window_size": w(7.0), "closed": c, "partial": partial});
                    add(&format!("wmean_{c}"), op("ewm_mean", "x", &h2, ex("keep")));
                    add(&format!("wvar_{c}"), op("ewm_var", "y", &h2, ex("drop")));
                    add(&format!("wsum_{c}"), op("ewm_sum", "x", &h5, ex("null")));
                    add(&format!("wrate_{c}"), op("ewm_rate", "y", &h5, ex("keep")));
                }
                add(
                    "wsum_inf",
                    op(
                        "ewm_sum",
                        "x",
                        &json!("inf"),
                        json!({"window_size": w(20.0), "min_samples": 4}),
                    ),
                );
            }
        }
        // Windows under one side of `closed` only, so that no kernel of the
        // other side drops the rows a `partial` of null or keep would show.
        8 | 9 => {
            let (c1, c2) = if flavor == 8 {
                ("right", "both")
            } else {
                ("left", "none")
            };
            let w = clk.t(7.0);
            for c in [c1, c2] {
                let ex = |partial: &str| json!({"window_size": w, "closed": c, "partial": partial});
                add(&format!("m_{c}"), op("ewm_mean", "x", &h2, ex("null")));
                add(&format!("s_{c}"), op("ewm_sum", "y", &h5, ex("null")));
                add(&format!("r_{c}"), op("ewm_rate", "x", &h5, ex("keep")));
                add(&format!("v_{c}"), op("ewm_var", "y", &h2, ex("keep")));
            }
        }
        // Quantiles: a sketch fed beside the queue, running and windowed.
        6 => {
            let w = clk.t(20.0);
            add(
                "q_running",
                op("ewm_quantile", "x", &h5, json!({"quantile": 0.5})),
            );
            add(
                "q_win",
                op(
                    "ewm_quantile",
                    "y",
                    &h2,
                    json!({"quantile": 0.25, "window_size": w, "closed": "left"}),
                ),
            );
            add("mean", op("ewm_mean", "x", &h5, json!({})));
        }
        // Forward kernels alone: every statistic and `closed`, `partial` and
        // `min_samples`, a window short and long, two inputs.
        10 | 11 => {
            let w = |secs: f64| clk.t(secs);
            let (c1, c2) = if flavor == 10 {
                ("right", "both")
            } else {
                ("left", "none")
            };
            for (c, secs) in [(c1, 7.0), (c2, 2.0)] {
                let ex = |partial: &str| json!({"window_size": w(secs), "closed": c, "partial": partial});
                add(&format!("fm_{c}"), op("rewm_mean", "x", &h2, ex("null")));
                add(&format!("fs_{c}"), op("rewm_sum", "y", &h5, ex("null")));
                add(&format!("fr_{c}"), op("rewm_rate", "x", &h5, ex("keep")));
                add(
                    &format!("fm2_{c}"),
                    op(
                        "rewm_mean",
                        "y",
                        &h5,
                        json!({"window_size": w(30.0), "closed": c, "min_samples": 3, "partial": "drop"}),
                    ),
                );
            }
        }
        // A forward quantile, a forward sum with no decay, and kernels of both
        // directions in one core, a plain one among them.
        12 | 13 => {
            let w = |secs: f64| clk.t(secs);
            add(
                "fq",
                op(
                    "rewm_quantile",
                    "x",
                    &h5,
                    json!({"quantile": 0.5, "window_size": w(20.0)}),
                ),
            );
            add(
                "fs_inf",
                op(
                    "rewm_sum",
                    "y",
                    &json!("inf"),
                    json!({"window_size": w(9.0), "closed": "both"}),
                ),
            );
            if flavor == 13 {
                add("mean", op("ewm_mean", "x", &h5, json!({})));
                add("sum", op("ewm_sum", "y", &h2, json!({"closed": "left"})));
                add(
                    "wmean",
                    op(
                        "ewm_mean",
                        "x",
                        &h2,
                        json!({"window_size": w(12.0), "closed": "both"}),
                    ),
                );
                add("var", op("ewm_var", "y", &h5, json!({})));
                add(
                    "ahead",
                    op("rewm_mean", "x", &h5, json!({"window_size": w(30.0)})),
                );
            }
        }
        // A forward kernel beside plain ones.
        _ => {
            let w = clk.t(30.0);
            add("mean", op("ewm_mean", "x", &h5, json!({})));
            add(
                "ahead",
                op("rewm_mean", "x", &h5, json!({"window_size": w})),
            );
        }
    }
    f
}

#[derive(Clone)]
struct Case {
    name: String,
    config: Value,
    clk: Clk,
    grouped: bool,
    session: bool,
    breaks: Breaks,
    keys: u32,
    /// The share of the rows the plain loop must have taken, a tenth.
    engages: u32,
    /// Whether the case is in the essentials tier.
    essential: bool,
}

fn cases() -> Vec<Case> {
    let mut cases = vec![];
    let regimes: [(&str, Breaks); 12] = [
        ("plain", Breaks::default()),
        (
            "nulls",
            Breaks {
                null: 0.02,
                ..Breaks::default()
            },
        ),
        (
            "rare nulls",
            Breaks {
                null: 0.002,
                ..Breaks::default()
            },
        ),
        (
            "ties",
            Breaks {
                tie: 0.01,
                ..Breaks::default()
            },
        ),
        (
            "many ties",
            Breaks {
                tie: 0.4,
                ..Breaks::default()
            },
        ),
        (
            "many nulls and ties",
            Breaks {
                null: 0.3,
                tie: 0.3,
                ..Breaks::default()
            },
        ),
        (
            "caps",
            Breaks {
                cap: 0.003,
                ..Breaks::default()
            },
        ),
        (
            "sessions",
            Breaks {
                session: 0.003,
                ..Breaks::default()
            },
        ),
        (
            "a silent key",
            Breaks {
                rare: 0.004,
                ..Breaks::default()
            },
        ),
        // A key whose rows are so far apart that most of them start it over.
        (
            "restarts",
            Breaks {
                rare: 0.012,
                ..Breaks::default()
            },
        ),
        // Steps at the edge of the cap.
        (
            "cap edge",
            Breaks {
                edge: 0.03,
                ..Breaks::default()
            },
        ),
        (
            "everything",
            Breaks {
                null: 0.004,
                tie: 0.004,
                cap: 0.002,
                session: 0.002,
                back: 0.001,
                rare: 0.003,
                edge: 0.002,
                ..Breaks::default()
            },
        ),
    ];
    for clk in [Clk::Datetime, Clk::Float, Clk::Int, Clk::Rows] {
        for flavor in 0..14 {
            for (rname, breaks) in regimes {
                // A row-count clock has no stamp to tie, no gap to cap and no
                // step back; a forward window under groups needs a clock column.
                if clk == Clk::Rows
                    && (breaks.tie > 0.0
                        || breaks.cap > 0.0
                        || breaks.back > 0.0
                        || breaks.edge > 0.0
                        || matches!(rname, "restarts"))
                {
                    continue;
                }
                // The clocks other than the datetime one take the same loop
                // with another step: a spread of flavors (running, every
                // `closed`, windows, a quantile, forward, mixed) keeps the
                // extended run to minutes.
                if clk != Clk::Datetime && !matches!(flavor, 0 | 1 | 2 | 5 | 6 | 10 | 13) {
                    continue;
                }
                let forward = matches!(flavor, 7 | 10..=13);
                for (grouped, keys) in [(false, 1), (true, 1), (true, 3), (true, 12), (true, 300)] {
                    if clk == Clk::Rows && grouped && forward {
                        continue;
                    }
                    // Many keys, every row of a key far from the one before: a
                    // few flavors and regimes, as each case is long.
                    if keys == 300
                        && !(matches!(flavor, 0 | 1 | 4 | 5 | 10 | 13)
                            && matches!(rname, "plain" | "nulls" | "everything" | "sessions"))
                    {
                        continue;
                    }
                    let session = breaks.session > 0.0;
                    let mut config = json!({"formulas": formulas(clk, flavor)});
                    if clk != Clk::Rows {
                        config["clock"] = json!("ts");
                        config["gap_cap"] = match clk {
                            Clk::Datetime => json!("30s"),
                            Clk::Int => json!(29999.5),
                            _ => json!(30.0),
                        };
                    }
                    if grouped {
                        config["group"] = json!("g");
                    }
                    if session {
                        config["session"] = json!("s");
                    }
                    if breaks.back > 0.0 {
                        config["restart_after_step_back"] = clk.t(1.0);
                    }
                    let many = keys == 300;
                    cases.push(Case {
                        name: format!(
                            "{} flavor {flavor} {rname} grouped {grouped} keys {keys}",
                            clk.name()
                        ),
                        config,
                        clk,
                        grouped,
                        session,
                        breaks,
                        keys,
                        // 300 keys: each key's first row is `push`'s, and with
                        // a session column each key's row after a change.
                        engages: if many { 3 } else { 8 },
                        essential: {
                            let few = matches!(keys, 1 | 12)
                                && matches!(
                                    rname,
                                    "plain" | "everything" | "many nulls and ties" | "a silent key"
                                )
                                && if clk == Clk::Datetime {
                                    matches!(
                                        flavor,
                                        0 | 1 | 2 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13
                                    )
                                } else {
                                    matches!(flavor, 0 | 2 | 5 | 13)
                                };
                            let many_keys = many
                                && clk != Clk::Float
                                && matches!(flavor, 0 | 5 | 13)
                                && rname == "plain";
                            (clk != Clk::Float && few) || many_keys
                        },
                    });
                }
            }
        }
    }
    cases.extend(review_cases());
    cases
}

/// Review 459, C7: what the cases above never held the plain loop to `push`
/// on: a `like=` spec's `accept` columns (a row with a null in one is
/// accepted by no fit), non-null NaN and infinities in a column with no
/// null, a cap no step reaches (a clocked run cannot leave `gap_cap` out),
/// and an integer clock at either end of an i64.
fn review_cases() -> Vec<Case> {
    let mut out = vec![];
    let mut add =
        |name: &str, clk: Clk, flavor: usize, breaks: Breaks, edit: &dyn Fn(&mut Value)| {
            for (grouped, keys) in [(false, 1), (true, 1), (true, 12)] {
                if clk == Clk::Rows && grouped && matches!(flavor, 7 | 10..=13) {
                    continue;
                }
                let mut config = json!({"formulas": formulas(clk, flavor)});
                if clk != Clk::Rows {
                    config["clock"] = json!("ts");
                    config["gap_cap"] = match clk {
                        Clk::Datetime => json!("30s"),
                        Clk::Int => json!(29999.5),
                        _ => json!(30.0),
                    };
                }
                if grouped {
                    config["group"] = json!("g");
                }
                edit(&mut config);
                out.push(Case {
                    name: format!(
                        "{} flavor {flavor} {name} grouped {grouped} keys {keys}",
                        clk.name()
                    ),
                    config,
                    clk,
                    grouped,
                    session: false,
                    breaks,
                    keys,
                    engages: 8,
                    // One flavor of each in the essentials (a stream fed a
                    // row at a time is long in a test build); all extended.
                    essential: clk != Clk::Float
                        && flavor
                            == match name {
                                "accept" | "nonfinite" => 1,
                                _ => 5,
                            },
                });
            }
        };
    let nulls = Breaks {
        null: 0.02,
        ..Breaks::default()
    };
    let accept = |c: &mut Value| c["like"] = json!({"spec": "fit", "accept": ["y"]});
    for flavor in [0, 1, 5, 13] {
        add("accept", Clk::Datetime, flavor, nulls, &accept);
    }
    add("accept", Clk::Rows, 1, nulls, &accept);
    let nonfinite = Breaks {
        nonfinite: 0.01,
        ..Breaks::default()
    };
    for clk in [Clk::Datetime, Clk::Int, Clk::Rows] {
        for flavor in [1, 5, 6] {
            add("nonfinite", clk, flavor, nonfinite, &|_| {});
        }
    }
    // A clocked run cannot leave `gap_cap` out (it is refused: the default,
    // an infinite cap, is the clockless run's, which the row-count cases
    // hold); a cap no step reaches is the nearest, every step under it.
    let far = |c: &mut Value| {
        c["gap_cap"] = match c["gap_cap"] {
            Value::String(_) => json!("100000d"),
            _ => json!(1e300),
        };
    };
    for clk in [Clk::Datetime, Clk::Float, Clk::Int] {
        for flavor in [0, 1, 5, 10, 13] {
            add(
                "a cap no step reaches",
                clk,
                flavor,
                Breaks::default(),
                &far,
            );
            add(
                "a cap no step reaches, long steps",
                clk,
                flavor,
                Breaks {
                    cap: 0.003,
                    ..Breaks::default()
                },
                &far,
            );
        }
    }
    for (name, base) in [
        ("i64 top", i64::MAX - 10_000_000_000),
        ("i64 bottom", i64::MIN + 1_000),
    ] {
        for flavor in [0, 1, 5, 10, 13] {
            add(
                name,
                Clk::Int,
                flavor,
                Breaks {
                    int_base: base,
                    ..Breaks::default()
                },
                &|_| {},
            );
        }
    }
    out
}

/// When a run saves its state and loads it into a fresh run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reload {
    Never,
    /// After the second chunk.
    Once,
    /// After every `n`th chunk.
    Every(usize),
}

/// One run of a case: its outputs and state, and how many rows the plain
/// loop took. The chunk lengths are taken in turn, again from the first once
/// they run out, until the stream is fed (review 459, C6: one-row chunks
/// covered the first 200 rows only, and the rest was never fed).
fn run(c: &Case, d: &Data, n: usize, chunks: &[usize], reload: Reload, plain: bool) -> (H, u64) {
    let config: WindowsConfig = serde_json::from_value(c.config.clone()).unwrap();
    let schema = frame(d, c.clk, c.grouped, c.session, 0, 1).schema().clone();
    let mut run = WindowsRun::new(config.clone(), &schema).unwrap();
    run.set_plain_run(plain);
    let mut h = H(0xcbf29ce484222325, 0);
    let (mut at, mut taken) = (0usize, 0u64);
    for (k, &len) in chunks.iter().cycle().enumerate() {
        let to = (at + len).min(n);
        if at >= to {
            break;
        }
        let f = frame(d, c.clk, c.grouped, c.session, at, to);
        match run.feed(&f, None) {
            Ok(o) => h.frame(&o),
            Err(e) => {
                // A refused step back: the same words either way.
                for b in format!("{e}").bytes() {
                    h.u(u64::from(b));
                }
                return (h, taken + run.plain_rows());
            }
        }
        at = to;
        let now = match reload {
            Reload::Never => false,
            Reload::Once => k == 1,
            Reload::Every(every) => (k + 1) % every == 0,
        };
        if now {
            let bytes = run.save_bytes().unwrap();
            h.bytes(&bytes);
            taken += run.plain_rows();
            run = WindowsRun::load_bytes(&bytes, config.clone(), &schema).unwrap();
            run.set_plain_run(plain);
            assert_eq!(run.save_bytes().unwrap(), bytes, "{} resave", c.name);
        }
    }
    h.frame(&run.finish().unwrap());
    h.bytes(&run.save_bytes().unwrap());
    (h, taken + run.plain_rows())
}

/// Every case `keep` takes, each stream `n` rows.
/// `small`: the stream fed a row at a time, and in chunks of 1, 2, 3 and 5
/// rows reloaded every 97th chunk (review 459, C6), the whole stream either
/// way; otherwise in one chunk, and in two chunkings reloaded once. Two
/// tests, so a test runner runs them side by side.
fn hold_to_push(keep: impl Fn(&Case) -> bool, n: usize, small: bool) {
    let mut engaged = 0u64;
    let mut total = 0u64;
    for (ci, c) in cases().iter().enumerate().filter(|(_, c)| keep(c)) {
        let d = data(n, 0x9E3779B97F4A7C15 ^ ci as u64, c.breaks, c.keys);
        let chunkings = if small {
            vec![
                (vec![1], Reload::Never),
                (vec![1, 2, 3, 5], Reload::Every(97)),
            ]
        } else {
            vec![
                (vec![n], Reload::Never),
                (vec![7, 1, 13, 64, 1500, 415, 3000], Reload::Once),
                (vec![1000, 1000, 1000], Reload::Once),
            ]
        };
        for (k, (chunks, reload)) in chunkings.iter().enumerate() {
            let reload = *reload;
            let (general, none) = run(c, &d, n, chunks, reload, false);
            let (plain, taken) = run(c, &d, n, chunks, reload, true);
            assert_eq!(none, 0, "{}: switched off, yet it took rows", c.name);
            assert_eq!(
                (plain.0, plain.1),
                (general.0, general.1),
                "{} chunking {k}: the plain loop moved a bit or a state byte",
                c.name
            );
            if !small && k == 0 && c.engages > 0 {
                // Most rows of a stream that is plain mostly.
                assert!(
                    taken * 10 >= (n as u64) * u64::from(c.engages),
                    "{}: the plain loop took {taken} of {n} rows",
                    c.name
                );
                engaged += 1;
            }
            total += taken;
        }
    }
    println!("plain loop engaged in {engaged} cases, {total} rows in all");
    assert!(small || engaged > 10);
    assert!(total > 0, "the plain loop took no row");
}

#[test]
fn the_plain_loop_gives_push_every_bit_and_every_state_byte() {
    hold_to_push(|c| c.essential, 1500, false);
}

#[test]
fn the_plain_loop_gives_push_every_bit_and_every_state_byte_a_few_rows_at_a_time() {
    hold_to_push(|c| c.essential, 1500, true);
}

#[test]
#[ignore = "extended: several minutes in a test build"]
fn the_plain_loop_gives_push_every_bit_and_every_state_byte_in_every_case() {
    hold_to_push(|_| true, 3000, false);
}

#[test]
#[ignore = "extended: several minutes in a test build"]
fn the_plain_loop_gives_push_every_bit_and_every_state_byte_in_every_case_a_few_rows_at_a_time() {
    hold_to_push(|_| true, 3000, true);
}
