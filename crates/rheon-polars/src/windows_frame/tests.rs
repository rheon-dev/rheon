//! The window run's unit tests, split from `windows_frame.rs` to keep it
//! under the repository's file-size cap.

use super::*;

fn config(json: &str) -> WindowsConfig {
    serde_json::from_str(json).unwrap()
}

const FORWARD: &str = r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
        {"half_life": 1, "window_size": 2}]}], "clock": "t", "gap_cap": 100}"#;

fn frame(t: &[f64]) -> DataFrame {
    df!(
        "t" => t,
        "x" => t.iter().map(|v| v * 10.0).collect::<Vec<_>>(),
        "wide" => t.iter().map(|v| format!("row {v} {}", "pad".repeat(20))).collect::<Vec<_>>(),
    )
    .unwrap()
}

/// Review P7: polars' IPC reader unwraps a field of a record batch's
/// header, so a state whose held rows were damaged there panicked at
/// the load rather than being refused. Every byte of the held rows'
/// IPC file, set to 0x00 or 0xff, is refused or loads; none panics.
///
/// Not every damage: the reader trusts the lengths the file states, and
/// one of this fixture's length bytes set to 0x7f asks it for 3.6e16
/// bytes, an allocation failure that aborts the process and that no
/// unwind can take. Only a check of the file's bytes before they are read, such
/// as a checksum in the state, would refuse that.
#[test]
fn damaged_held_rows_are_refused_and_never_panic() {
    let cfg = config(FORWARD);
    let f = frame(&[1.0, 2.0, 3.0, 4.0]);
    let mut run = WindowsRun::new(cfg.clone(), f.schema()).unwrap();
    run.feed(&f, None).unwrap();
    assert!(run.held() > 0, "the fixture: rows held");
    let good = run.save_bytes().unwrap();
    let magic = b"ARROW1";
    let at = |from: usize| {
        good[from..]
            .windows(magic.len())
            .position(|w| w == magic)
            .map(|i| from + i)
    };
    let start = at(0).expect("the held rows' IPC file");
    let end = at(start + magic.len()).expect("its closing magic") + magic.len();
    let mut refused = 0;
    for i in start..end {
        for v in [0x00u8, 0xff] {
            if good[i] == v {
                continue;
            }
            let mut bad = good.clone();
            bad[i] = v;
            let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                WindowsRun::load_bytes(&bad, cfg.clone(), f.schema()).map(drop)
            }));
            match got {
                Ok(Ok(())) => {}
                Ok(Err(_)) => refused += 1,
                Err(_) => panic!("byte {i} set to {v:#04x} panicked the load"),
            }
        }
    }
    assert!(refused > 0, "some damage is refused");
}

/// The cores of one spec's groups hold one copy of the call and its plan,
/// a core of another input does not take them, and sharing changes
/// nothing a core saves.
#[test]
fn runs_of_one_call_share_their_constants_and_only_those() {
    let cfg = config(FORWARD);
    let f = frame(&[1.0, 2.0, 3.0]);
    let mut a = WindowsRun::new(cfg.clone(), f.schema()).unwrap();
    let mut b = WindowsRun::new(cfg.clone(), f.schema()).unwrap();
    a.feed(&f, None).unwrap();
    let unshared = (a.save_bytes().unwrap(), b.save_bytes().unwrap());
    assert!(!Arc::ptr_eq(&a.k, &b.k));
    b.share_with(&a);
    assert!(Arc::ptr_eq(&a.k, &b.k));
    assert!(a.core.shares_definitions_with(&b.core));
    assert_eq!((a.save_bytes().unwrap(), b.save_bytes().unwrap()), unshared);
    // A copy shares them too, and feeding it leaves the original alone.
    let mut c = a.clone();
    assert!(Arc::ptr_eq(&a.k, &c.k));
    c.feed(&frame(&[4.0]), None).unwrap();
    assert_eq!(a.save_bytes().unwrap(), unshared.0);
    // Another input's schema is another plan.
    let wider = df!("t" => [1.0], "x" => [1i64], "wide" => ["a"]).unwrap();
    let mut d = WindowsRun::new(cfg, wider.schema()).unwrap();
    d.share_with(&a);
    assert!(!Arc::ptr_eq(&a.k, &d.k));
}

/// A bare column input skips the lazy engine's cast and a bare operator
/// formula its plan, and the bits are the lazy engine's: every value, null
/// and row number of what a run resolves, over nulls, NaN, infinities,
/// signed zeros and a column of integers (which takes the plan), whole and
/// in chunks, for a mix of formulas that take the short cut and ones that
/// do not.
#[test]
fn the_short_cuts_resolve_as_the_lazy_engine_does() {
    let op = |kind: &str, col: &str, extra: &str| {
        format!(r#"["{kind}", ["col", "{col}"], {{"half_life": 1.5, "window_size": 4{extra}}}]"#)
    };
    let shapes: Vec<(&str, Vec<String>)> = vec![
            ("bare forward", vec![op("rewm_sum", "x", "")]),
            (
                "bare drop",
                vec![op("rewm_mean", "x", r#", "partial": "drop""#)],
            ),
            (
                "bare, aliased, two inputs",
                vec![
                    format!(r#"["alias", {}, "a"]"#, op("rewm_mean", "x", "")),
                    op("ewm_sum", "y", ""),
                ],
            ),
            (
                "one not bare",
                vec![
                    op("rewm_sum", "x", ""),
                    format!(r#"["-", {}, ["col", "x"]]"#, op("ewm_mean", "x", "")),
                ],
            ),
            ("integer input", vec![op("rewm_sum", "n", "")]),
            (
                "computed input",
                vec![
                    r#"["rewm_sum", ["*", ["col", "x"], ["col", "y"]], {"half_life": 1.5, "window_size": 4}]"#
                        .to_string(),
                ],
            ),
        ];
    let n = 60;
    let mut seed = 7u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut t = 0.0;
    let (mut ts, mut xs, mut ys, mut ns) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        t += if i % 9 == 4 { 0.0 } else { 0.2 + 1.5 * next() };
        ts.push(t);
        let v = next();
        xs.push(match i % 11 {
            3 => None,
            5 => Some(f64::NAN),
            7 => Some(f64::INFINITY),
            8 => Some(-0.0),
            _ => Some(v - 0.5),
        });
        ys.push(Some(next() * 3.0));
        ns.push(i as i64 % 5);
    }
    let rows: Vec<u64> = (0..n as u64).collect();
    let df = df!("t" => ts, "x" => xs, "y" => ys, "n" => ns, "@po:row" => rows.clone()).unwrap();
    let run = |formulas: &[String], cuts: &[usize], off: bool| -> Vec<DataFrame> {
        NO_SHORT_CUTS.with(|c| c.set(off));
        let names: Vec<String> = (0..formulas.len())
            .map(|i| format!(r#"{{"name": "f{i}", "tree": {}}}"#, formulas[i]))
            .collect();
        let mut r = WindowsRun::new(
            config(&format!(
                r#"{{"formulas": [{}], "clock": "t", "gap_cap": 100}}"#,
                names.join(",")
            )),
            df.schema(),
        )
        .unwrap();
        let mut out = vec![];
        let mut at = 0;
        for &c in cuts {
            let part = df.slice(at as i64, c);
            out.push(r.feed_resolving(&part, &rows[at..at + c]).unwrap());
            at += c;
        }
        NO_SHORT_CUTS.with(|c| c.set(false));
        out
    };
    let bits = |frames: &[DataFrame]| -> Vec<String> {
        frames
            .iter()
            .flat_map(|f| {
                f.columns().iter().map(|c| match c.dtype() {
                    DataType::Float64 => format!(
                        "{} {:?}",
                        c.name(),
                        c.f64()
                            .unwrap()
                            .iter()
                            .map(|v| v.map(f64::to_bits))
                            .collect::<Vec<_>>()
                    ),
                    _ => format!(
                        "{} {:?} {:?}",
                        c.name(),
                        c.dtype(),
                        c.u64().map(|u| u.iter().collect::<Vec<_>>())
                    ),
                })
            })
            .collect()
    };
    for (what, formulas) in &shapes {
        for cuts in [vec![n], vec![7, 1, 30, 22]] {
            let fast = run(formulas, &cuts, false);
            let lazy = run(formulas, &cuts, true);
            let resolved: usize = lazy.iter().map(DataFrame::height).sum();
            assert!(resolved > 20, "{what}: {resolved} rows resolved");
            assert_eq!(bits(&fast), bits(&lazy), "{what}, cuts {cuts:?}");
        }
    }
}

/// `feed` over bare operators is the lazy engine's: the same columns in
/// the same order, with the same nulls and bits, for inputs with and
/// without a null, in one piece or several, sliced, with an extra column
/// the formulas do not read, an operator partial under `"drop"`, and
/// chunks that cut anywhere (the rows' dropped and not).
#[test]
fn a_feed_of_bare_operators_is_the_lazy_engines() {
    let op = |kind: &str, col: &str, extra: &str| {
        format!(r#"["{kind}", ["col", "{col}"], {{"half_life": 1.5, "window_size": 4{extra}}}]"#)
    };
    let shapes: Vec<(&str, Vec<String>)> = vec![
        ("one", vec![op("ewm_mean", "x", "")]),
        ("drop", vec![op("ewm_mean", "x", r#", "partial": "drop""#)]),
        (
            "aliased, two inputs, one shared",
            vec![
                format!(r#"["alias", {}, "a"]"#, op("ewm_sum", "x", "")),
                op("ewm_mean", "y", ""),
                op("ewm_sum", "x", ""),
            ],
        ),
        (
            "no window",
            vec![r#"["ewm_mean", ["col", "y"], {"half_life": 1.5}]"#.into()],
        ),
        ("forward", vec![op("rewm_sum", "x", "")]),
        (
            "one not bare",
            vec![
                op("ewm_sum", "x", ""),
                format!(r#"["-", {}, ["col", "x"]]"#, op("ewm_mean", "x", "")),
            ],
        ),
    ];
    let n = 70;
    let mut seed = 11u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut t = 0.0;
    let (mut ts, mut xs, mut ys, mut zs) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        t += if i % 9 == 4 { 0.0 } else { 0.2 + 1.5 * next() };
        ts.push(t);
        xs.push(match i % 11 {
            3 => None,
            5 => Some(f64::NAN),
            _ => Some(next() - 0.5),
        });
        ys.push(Some(next() * 3.0));
        zs.push(i as i64);
    }
    let df = df!("t" => ts, "x" => xs, "y" => ys, "z" => zs).unwrap();
    // The same rows as columns of several pieces.
    let pieces = {
        let mut whole = df.slice(0, 23);
        whole.vstack_mut(&df.slice(23, 30)).unwrap();
        whole.vstack_mut(&df.slice(53, 17)).unwrap();
        whole
    };
    let run = |formulas: &[String], frames: &[DataFrame], off: bool| -> Vec<DataFrame> {
        NO_SHORT_CUTS.with(|c| c.set(off));
        let names: Vec<String> = (0..formulas.len())
            .map(|i| format!(r#"{{"name": "f{i}", "tree": {}}}"#, formulas[i]))
            .collect();
        let mut r = WindowsRun::new(
            config(&format!(
                r#"{{"formulas": [{}], "clock": "t", "gap_cap": 100}}"#,
                names.join(",")
            )),
            df.schema(),
        )
        .unwrap();
        let mut out: Vec<DataFrame> = frames.iter().map(|f| r.feed(f, None).unwrap()).collect();
        out.push(r.finish().unwrap());
        NO_SHORT_CUTS.with(|c| c.set(false));
        out
    };
    let bits = |frames: &[DataFrame]| -> Vec<String> {
        frames
            .iter()
            .flat_map(|f| {
                f.columns().iter().map(|c| match c.dtype() {
                    DataType::Float64 => format!(
                        "{} {:?}",
                        c.name(),
                        c.f64()
                            .unwrap()
                            .iter()
                            .map(|v| v.map(f64::to_bits))
                            .collect::<Vec<_>>()
                    ),
                    d => format!("{} {d:?} {}", c.name(), c.len()),
                })
            })
            .collect()
    };
    let cuts = |sizes: &[usize]| -> Vec<DataFrame> {
        let mut at = 0;
        sizes
            .iter()
            .map(|&c| {
                let part = df.slice(at as i64, c);
                at += c;
                part
            })
            .collect()
    };
    for (what, formulas) in &shapes {
        for (how, frames) in [
            ("whole", vec![df.clone()]),
            ("pieces", vec![pieces.clone()]),
            ("cut", cuts(&[7, 1, 30, 32])),
            ("one rows", cuts(&[1; 70])),
        ] {
            let fast = run(formulas, &frames, false);
            let lazy = run(formulas, &frames, true);
            let rows: usize = lazy.iter().map(DataFrame::height).sum();
            assert!(rows > 30, "{what}, {how}: {rows} rows out");
            assert_eq!(bits(&fast), bits(&lazy), "{what}, {how}");
            let names: Vec<&str> = fast[0]
                .get_column_names()
                .iter()
                .map(|n| n.as_str())
                .collect();
            assert!(names.contains(&"f0") && names.contains(&"t"), "{names:?}");
        }
    }
}

/// An operator's values as a column: null where the value is NaN, the
/// rest to the bit, a null's slot 0 (as a column built from options leaves
/// it), with and without a NaN.
#[test]
fn an_operators_values_become_a_column_with_a_null_for_each_nan() {
    for values in [
        vec![],
        vec![1.5, -0.0, 3.25],
        vec![f64::NAN, f64::NAN],
        vec![f64::NAN, 2.0, f64::NAN, -4.5, f64::NAN],
        (0..130)
            .map(|i| if i % 3 == 0 { f64::NAN } else { f64::from(i) })
            .collect(),
    ] {
        let column = nullable_column("v", values.clone());
        let want: Float64Chunked = values.iter().map(|&x| (!x.is_nan()).then_some(x)).collect();
        let got = column.f64().unwrap();
        assert_eq!(got.name().as_str(), "v");
        assert_eq!(got.null_count(), want.null_count());
        assert_eq!(
            got.iter().map(|v| v.map(f64::to_bits)).collect::<Vec<_>>(),
            want.iter().map(|v| v.map(f64::to_bits)).collect::<Vec<_>>()
        );
        // The slot under a null reads 0, as the options' column leaves it.
        let raw: Vec<u64> = got
            .downcast_iter()
            .flat_map(|a| a.values().iter().map(|x| x.to_bits()))
            .collect();
        let raw_want: Vec<u64> = want
            .downcast_iter()
            .flat_map(|a| a.values().iter().map(|x| x.to_bits()))
            .collect();
        assert_eq!(raw, raw_want);
    }
}

/// What a chunk's groups share, evaluated once ([`WindowsRun::share`]),
/// is what each group's run computes for itself: runs fed their rows with
/// the chunk's shared inputs give the same rows to the bit and keep the
/// same state as runs that plan their own, over two chunks (the second
/// starts from the first's increments), with a null, a step back past a
/// restart, a session change and an integer input; and the shared values
/// are there, not a fallback.
#[test]
fn a_chunk_s_shared_increments_are_each_run_s_own() {
    let n = 90;
    let key = |i: usize| (i * 7 + i / 5) % 4;
    let frame_of = |lo: usize, hi: usize| {
        let rows = lo..hi;
        df!(
                "t" => rows.clone().map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
                "tr" => rows.clone().map(|i| if i == 50 { 3.0 } else { i as f64 * 0.5 }).collect::<Vec<_>>(),
                "x" => rows.clone().map(|i| (i % 11 != 4).then(|| (i * i % 17) as f64 * 0.25)).collect::<Vec<_>>(),
                "k" => rows.clone().map(|i| (i % 9 != 2).then_some((i * 3 % 13) as i64)).collect::<Vec<_>>(),
                "s" => rows.clone().map(|i| if i < 40 { "a" } else { "b" }).collect::<Vec<_>>(),
                "g" => rows.clone().map(|i| key(i) as i64).collect::<Vec<_>>(),
                "@po:row" => rows.map(|i| i as u64).collect::<Vec<_>>(),
            )
            .unwrap()
    };
    let window = r#"{"half_life": 4, "window_size": 6}"#;
    for (what, inputs_shared, tree, policy) in [
        (
            "bare",
            true,
            format!(r#"["rewm_mean", ["increment", ["col", "x"]], {window}]"#),
            r#", "clock": "t", "gap_cap": 100"#,
        ),
        (
            "arithmetic and two increments",
            true,
            format!(
                r#"["-", ["rewm_sum", ["*", ["increment", ["col", "x"]], ["col", "x"]], {window}],
                    ["rewm_mean", ["increment", ["col", "k"]], {window}]]"#
            ),
            r#", "clock": "t", "gap_cap": 100"#,
        ),
        (
            // Every input is a column of doubles: each run reads its own.
            "an increment in the formula's body",
            false,
            format!(r#"["-", ["rewm_mean", ["col", "x"], {window}], ["increment", ["col", "x"]]]"#),
            r#", "clock": "t", "gap_cap": 100"#,
        ),
        (
            "a restart",
            true,
            format!(r#"["rewm_mean", ["increment", ["col", "x"]], {window}]"#),
            r#", "clock": "tr", "gap_cap": 100, "restart_after_step_back": 0"#,
        ),
        (
            "a session",
            true,
            format!(r#"["rewm_mean", ["increment", ["col", "k"]], {window}]"#),
            r#", "clock": "t", "gap_cap": 100, "session": "s", "session_gap": "reset""#,
        ),
        (
            // A tick sign beside an increment, each with its
            // own slot, and the sign carried past them.
            "a tick sign",
            true,
            format!(
                r#"["-", ["rewm_sum", ["*", ["tick_sign", ["col", "k"]], ["col", "x"]], {window}],
                    ["rewm_mean", ["increment", ["col", "x"]], {window}]]"#
            ),
            r#", "clock": "t", "gap_cap": 100, "session": "s", "session_gap": "reset""#,
        ),
    ] {
        let json = format!(r#"{{"formulas": [{{"name": "f", "tree": {tree}}}]{policy}}}"#);
        let first = frame_of(0, 35);
        let schema = first.schema().clone();
        let new_runs = || -> Vec<WindowsRun> {
            (0..4)
                .map(|_| WindowsRun::new(config(&json), &schema).unwrap())
                .collect()
        };
        let (mut own, mut shared) = (new_runs(), new_runs());
        let bits = |df: &DataFrame| -> Vec<(String, Vec<Option<u64>>)> {
            df.columns()
                .iter()
                .map(|c| {
                    let cast = c.cast(&DataType::Float64).unwrap();
                    let values = cast.f64().unwrap().iter().map(|v| v.map(f64::to_bits));
                    (c.name().to_string(), values.collect())
                })
                .collect()
        };
        let mut resolved = 0;
        for (lo, hi) in [(0, 35), (35, n)] {
            let chunk = frame_of(lo, hi);
            let rows_of: Vec<Vec<usize>> = (0..4)
                .map(|g| (0..chunk.height()).filter(|&r| key(lo + r) == g).collect())
                .collect();
            let groups: Vec<&[usize]> = rows_of.iter().map(Vec::as_slice).collect();
            let all = WindowsRun::share(&shared, &chunk, &groups)
                .expect("a chunk of several groups shares its inputs");
            assert!(
                all.incr.is_some(),
                "{what}: the increments' inputs are shared"
            );
            assert_eq!(all.inputs.is_some(), inputs_shared, "{what}: the inputs");
            for (g, rows) in rows_of.iter().enumerate() {
                let take = IdxCa::from_vec("".into(), rows.iter().map(|&r| r as IdxSize).collect());
                let sub = chunk.take(&take).unwrap();
                let numbers: Vec<u64> = rows.iter().map(|&r| (lo + r) as u64).collect();
                let want = own[g].feed_resolving(&sub, &numbers).unwrap();
                let mut got = shared[g]
                    .feed_resolving_with(&sub, &numbers, Some(all.of_group(rows)))
                    .unwrap();
                if shared[g].formulas_pending() {
                    got = shared[g].resolved_formulas(got).unwrap();
                }
                resolved += want.height();
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{what}: group {g}, rows {lo}..{hi}"
                );
            }
        }
        assert!(resolved > 40, "{what}: {resolved} rows resolved");
        for (g, (a, b)) in own.iter().zip(&shared).enumerate() {
            assert_eq!(
                a.save_bytes().unwrap(),
                b.save_bytes().unwrap(),
                "{what}: group {g}'s state"
            );
        }
        // Without an increment, the operator inputs alone.
        let plain = format!(
            r#"{{"formulas": [{{"name": "f", "tree": ["rewm_sum", ["*", ["col", "x"], ["col", "x"]], {window}]}}]{policy}}}"#
        );
        let plain = vec![WindowsRun::new(config(&plain), &schema).unwrap(); 4];
        let rows20: Vec<Vec<usize>> = (0..4)
            .map(|g| (0..20).filter(|&r| key(r) == g).collect())
            .collect();
        let groups20: Vec<&[usize]> = rows20.iter().map(Vec::as_slice).collect();
        let only = WindowsRun::share(&plain, &frame_of(0, 20), &groups20).unwrap();
        assert!(only.inputs.is_some() && only.incr.is_none(), "{what}");
        // Nothing is shared for a run that skips rows of its input, or
        // one keyed by a group column: each plans for itself.
        let chunk = frame_of(0, 20);
        let rows_of: Vec<Vec<usize>> = (0..4)
            .map(|g| (0..20).filter(|&r| key(r) == g).collect())
            .collect();
        let groups: Vec<&[usize]> = rows_of.iter().map(Vec::as_slice).collect();
        let mut skipping = new_runs();
        skipping[3].resume_skip = 2;
        assert!(WindowsRun::share(&skipping, &chunk, &groups).is_none());
        let grouped = r#", "group": "g""#;
        let json = format!(r#"{{"formulas": [{{"name": "f", "tree": {tree}}}]{policy}{grouped}}}"#);
        let keyed = vec![WindowsRun::new(config(&json), &schema).unwrap(); 4];
        assert!(WindowsRun::share(&keyed, &chunk, &groups).is_none());
        // A part that is not one value a row for each input and each
        // row is refused, whichever of the two is the wrong one.
        let all = WindowsRun::share(&new_runs(), &chunk, &groups).unwrap();
        let sub = chunk
            .take(&IdxCa::from_vec(
                "".into(),
                rows_of[0].iter().map(|&r| r as IdxSize).collect(),
            ))
            .unwrap();
        let numbers: Vec<u64> = rows_of[0].iter().map(|&r| r as u64).collect();
        let (whole, short) = (all.of_group(&rows_of[0]), all.of_group(&rows_of[0][1..]));
        let wrong_incr = Given {
            inputs: whole.inputs.clone(),
            incr: short.incr.clone(),
        };
        let wrong_inputs = Given {
            inputs: short.inputs.clone(),
            incr: whole.incr.clone(),
        };
        for (which, given) in [("increments", wrong_incr), ("inputs", wrong_inputs)] {
            if given.inputs.is_none() && which == "inputs" {
                continue;
            }
            let err = new_runs()[0]
                .feed_resolving_with(&sub, &numbers, Some(given))
                .err()
                .unwrap_or_else(|| panic!("{what}: wrong {which} accepted"));
            assert!(
                err.to_string().contains("internal: feed_resolving"),
                "{err}"
            );
        }
    }
}

/// Task 160, PB2: without a clock a row is one step, and a cap on it cut
/// every window at every row (a gap of 1 past a cap of 0.5).
#[test]
fn gap_cap_needs_a_clock() {
    let input = frame(&[0.0, 1.0]).schema().clone();
    let err = WindowsRun::new(
        config(
            r#"{"formulas": [{"name": "f", "tree": ["rewm_sum", ["col", "x"],
                {"half_life": 10, "window_size": 3}]}], "gap_cap": 0.5}"#,
        ),
        &input,
    )
    .err()
    .expect("refused");
    assert!(err.contains("with_windows: gap_cap needs clock"), "{err}");
}

/// Task 278 (D3): a `session_gap` without a session, in either form,
/// and a `restart_after_step_back` without a clock did nothing and were
/// taken; a spec refuses both, and so does this call now, in its own
/// words.
#[test]
fn a_key_with_nothing_to_act_on_is_refused() {
    let input = df!("t" => [0.0, 1.0], "x" => [1.0, 2.0], "s" => ["a", "b"])
        .unwrap()
        .schema()
        .clone();
    let refused = |extra: &str| {
        WindowsRun::new(
            config(&format!(
                r#"{{"formulas": [{{"name": "f", "tree": ["ewm_mean", ["col", "x"],
                    {{"half_life": 10, "window_size": 3}}]}}]{extra}}}"#
            )),
            &input,
        )
        .err()
        .expect("refused")
    };
    let gap = "with_windows: session_gap is set, but session is not, and session_gap says \
                   only what a change of session does. Set session to the session column, or \
                   leave out session_gap.";
    for extra in [
        r#", "session_gap": 1"#,
        r#", "session_gap": "reset""#,
        r#", "clock": "t", "gap_cap": 5, "session_gap": 1"#,
    ] {
        assert_eq!(refused(extra), gap, "{extra}");
    }
    for extra in [
        r#", "restart_after_step_back": 1"#,
        r#", "restart_after_step_back": 0, "session": "s""#,
    ] {
        assert_eq!(
            refused(extra),
            "with_windows: restart_after_step_back is set, but clock is not, and without a \
                 clock column every row is one step forward, so no row steps back. Set clock to \
                 the clock column, or leave out restart_after_step_back.",
            "{extra}"
        );
    }
}

/// Every window ends at a session change, so no window measures the
/// clock's jump across one, and a numeric `session_gap` is never read:
/// the call is refused naming it, as Python's is, and `"reset"` is still
/// taken. A state saved by 1.x with a number resumes the call without
/// one (the comparison of a saved call drops the number).
#[test]
fn a_numeric_session_gap_is_refused() {
    const OPS: &str = r#"{"formulas": [
            {"name": "back", "tree": ["ewm_mean", ["col", "x"], {"half_life": 2, "window_size": 3}]}],
            "clock": "t", "gap_cap": 4, "session": "s""#;
    let call = |gap: &str| config(&format!("{OPS}{gap}}}"));
    let df = df!(
        "t" => [0.0, 1.0, 2.0, 3.0],
        "x" => [1.0, 3.0, 2.0, 5.0],
        "s" => ["a", "a", "b", "b"],
    )
    .unwrap();
    for gap in [0, 1, 50] {
        let err = call(&format!(r#", "session_gap": {gap}"#))
            .refuse_unread_session_gap()
            .expect_err("refused");
        assert!(
            err.contains(&format!("with_windows: session_gap = {gap}"))
                && err.contains("is no longer read"),
            "{err}"
        );
    }
    for ok in [r#", "session_gap": "reset""#, ""] {
        call(ok).refuse_unread_session_gap().unwrap();
        WindowsRun::new(call(ok), df.schema()).unwrap();
    }
    // Under `like=` the spec's own is lent, and the run is built.
    let mut lent = call(r#", "session_gap": 1"#);
    lent.like = Some(Like {
        spec: "m".into(),
        accept: vec!["x".into()],
    });
    lent.refuse_unread_session_gap().unwrap();
}

/// Task 173, PC1: without a clock column each group's clock counts its
/// own rows, so a group that falls silent never closed its windows, and
/// every later row of every group waited for them to the end of the
/// input (500,001 rows held, 117 MiB more over 2M rows). Refused, naming
/// a clock column or no group; the same call with a clock runs, and so
/// do a window looking back under groups and one looking ahead without.
#[test]
fn a_window_looking_ahead_with_group_needs_a_clock() {
    let df = df!(
        "t" => [0.0, 1.0, 2.0, 3.0],
        "g" => ["a", "b", "b", "b"],
        "x" => [1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let ahead = r#"[{"name": "f", "tree": ["-", ["rewm_sum", ["col", "x"],
            {"half_life": 10, "window_size": 2}], ["col", "x"]]}]"#;
    let back = r#"[{"name": "f", "tree": ["ewm_sum", ["col", "x"], {"half_life": 10}]}]"#;
    let run = |formulas: &str, policy: &str| {
        WindowsRun::new(
            config(&format!(r#"{{"formulas": {formulas}{policy}}}"#)),
            df.schema(),
        )
    };
    let err = run(ahead, r#", "group": "g""#).err().expect("refused");
    assert!(
        err.starts_with("with_windows: a window looking ahead under group needs clock")
            && err.ends_with("Set clock and gap_cap, or leave group out."),
        "{err}"
    );
    // `like=` takes a spec's policy, and is held to the same rule.
    let like = r#", "group": "g", "like": {"spec": "m", "accept": ["x"]}"#;
    let err = run(ahead, like).err().expect("refused");
    assert!(err.contains("under group needs clock"), "{err}");
    let mut clocked = run(ahead, r#", "group": "g", "clock": "t", "gap_cap": 1.5"#).unwrap();
    let mut out = clocked.feed(&df, None).unwrap();
    out.vstack_mut(&clocked.finish().unwrap()).unwrap();
    assert_eq!(out.height(), 4);
    run(back, r#", "group": "g""#).unwrap();
    run(ahead, "").unwrap();
}

/// Task 173, PC2: a spec's window target runs the same core, through
/// `feed_resolving`, and a reset keeps the window whose far edge is the
/// last row before it whole there too: row 1's window `(1, 3]` resolves
/// at the step back with `4 + 8`, where it resolved discarded, null.
/// The windows reaching past that row resolve discarded, null.
#[test]
fn a_target_window_whose_far_edge_is_the_last_row_before_a_reset_resolves_whole() {
    let df = df!(
        "t" => [0.0, 1.0, 2.0, 3.0, -7.0, -6.0],
        "x" => [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        "@po:row" => [100u64, 101, 102, 103, 104, 105],
    )
    .unwrap();
    let mut run = WindowsRun::new(
        config(
            r#"{"formulas": [{"name": "f", "tree": ["rewm_sum", ["col", "x"],
                {"half_life": "inf", "window_size": 2}]}], "clock": "t", "gap_cap": 100,
                "restart_after_step_back": 1}"#,
        ),
        df.schema(),
    )
    .unwrap();
    let out = run
        .feed_resolving(&df, &[100, 101, 102, 103, 104, 105])
        .unwrap();
    let rows = |name: &str| -> Vec<u64> {
        out.column(name)
            .unwrap()
            .u64()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    };
    let f: Vec<Option<f64>> = out.column("f").unwrap().f64().unwrap().iter().collect();
    assert_eq!(rows("@po:seq"), [100, 101, 102, 103]);
    assert_eq!(rows("@po:at"), [103, 104, 104, 104]);
    assert_eq!(f, [Some(6.0), Some(12.0), None, None]);
}

/// Task 160, PC3: a state compares its call by length, not by how a
/// length was written: "5000ms" resumes the call that saved "5s", and
/// the rows it holds go on to the same values.
#[test]
fn a_state_resumes_another_spelling_of_one_length() {
    let df = df!(
        "t" => Series::new("t".into(), [0i64, 1_000, 2_000, 3_000, 4_000, 5_000])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap(),
        "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )
    .unwrap();
    let call = |cap: &str, window: &str| {
        config(&format!(
            r#"{{"formulas": [{{"name": "f", "tree": ["rewm_sum", ["col", "x"],
                {{"half_life": "1h", "window_size": "{window}"}}]}}], "clock": "t",
                "gap_cap": "{cap}"}}"#
        ))
    };
    let mut whole = WindowsRun::new(call("5s", "2s"), df.schema()).unwrap();
    let mut want = whole.feed(&df, None).unwrap();
    want.vstack_mut(&whole.finish().unwrap()).unwrap();
    let mut first = WindowsRun::new(call("5s", "2s"), df.schema()).unwrap();
    let got = first.feed(&df.slice(0, 3), None).unwrap();
    let bytes = first.save_bytes().unwrap();
    for (cap, window) in [("5000ms", "2s"), ("5s", "2000ms")] {
        let mut second = WindowsRun::load_bytes(&bytes, call(cap, window), df.schema())
            .unwrap_or_else(|e| panic!("{cap}, {window}: {e}"));
        let mut out = got.clone();
        out.vstack_mut(&second.feed(&df.slice(3, 3), None).unwrap())
            .unwrap();
        out.vstack_mut(&second.finish().unwrap()).unwrap();
        assert!(out.equals_missing(&want), "{cap}, {window}");
    }
    assert!(
        WindowsRun::load_bytes(&bytes, call("6s", "2s"), df.schema())
            .err()
            .expect("another length is another call")
            .contains("another call")
    );
}

fn values_ptr(df: &DataFrame, c: &str) -> *const f64 {
    df.column(c)
        .unwrap()
        .f64()
        .unwrap()
        .cont_slice()
        .unwrap()
        .as_ptr()
}

/// The rows a look-ahead holds are the input's own chunks, not copies:
/// the output's values sit at the input's addresses.
/// Review R1, N7 and N8: a formula with no operator beside one with an
/// operator was accepted (Python refused it), and an input under the
/// operators' prefix would have collided with a hidden column at run
/// time.
#[test]
fn each_formula_holds_an_operator_and_reads_no_reserved_column() {
    let input = frame(&[0.0, 1.0]).schema().clone();
    let err = WindowsRun::new(
        config(
            r#"{"formulas": [{"name": "y", "tree": ["ewm_sum", ["col", "x"], {"half_life": 1.0}]},
                                 {"name": "z", "tree": ["-", ["col", "x"], ["lit", 1.0]]}],
                   "clock": "t", "gap_cap": 100.0}"#,
        ),
        &input,
    )
    .err()
    .expect("refused");
    assert!(
        err.contains("holds no window operator or increment"),
        "{err}"
    );
    let mut with_hidden: Schema = (*input).clone();
    with_hidden.insert("@po:x".into(), DataType::Float64);
    let err = WindowsRun::new(
            config(
                r#"{"formulas": [{"name": "y", "tree": ["ewm_sum", ["col", "@po:x"], {"half_life": 1.0}]}],
                   "clock": "t", "gap_cap": 100.0}"#,
            ),
            &with_hidden,
        )
        .err()
        .expect("refused");
    assert!(
        err.contains("is kept for the window run's own columns"),
        "{err}"
    );
}

#[test]
fn held_rows_are_the_inputs_own_chunks() {
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    let out = run.feed(&df, None).unwrap();
    assert_eq!(out.height(), 3, "rows 0..3 resolved by row 5");
    assert_eq!(values_ptr(&out, "x"), values_ptr(&df, "x"));
    assert_eq!(run.held(), 3);
    let rest = run.finish().unwrap();
    assert_eq!(rest.height(), 3);
    // Row 3's window (3, 5] has its far edge on the last row: whole.
    // The windows of rows 4 and 5 reach past the input: null.
    let f: Vec<Option<f64>> = rest.column("f").unwrap().f64().unwrap().iter().collect();
    assert_eq!(f, [Some(40.0), None, None]);
}

/// The schema is the input's columns then the formulas, typed by what
/// they compute.
#[test]
fn the_schema_types_each_formula() {
    let df = frame(&[0.0, 1.0]);
    let cfg = config(
        r#"{"formulas": [
                {"name": "m", "tree": ["ewm_mean", ["col", "x"], {"half_life": 1}]},
                {"name": "up", "tree": [">", ["ewm_sum", ["col", "x"], {"half_life": 1}], ["lit", 0]]},
                {"name": "dx", "tree": ["increment", ["col", "x"]]}
            ], "clock": "t", "gap_cap": 100}"#,
    );
    let run = WindowsRun::new(cfg, df.schema()).unwrap();
    let s = run.schema();
    assert_eq!(
        s.iter_names().map(|n| n.to_string()).collect::<Vec<_>>(),
        ["t", "x", "wide", "m", "up", "dx"]
    );
    assert_eq!(s.get("up"), Some(&DataType::Boolean));
    assert_eq!(s.get("dx"), Some(&DataType::Float64));
    assert_eq!(run.needed(), vec!["t".to_string(), "x".to_string()]);
}

/// A formula that cannot run is refused while the plan is built, by
/// name.
#[test]
fn a_formula_that_cannot_run_is_refused_by_name() {
    let df = frame(&[0.0, 1.0]);
    for (json, says) in [
        (
            r#"{"formulas": [], "clock": "t", "gap_cap": 1}"#,
            "was given. Give at least one",
        ),
        (
            r#"{"formulas": [{"name": "x", "tree": ["ewm_mean", ["col", "x"], {"half_life": 1}]}], "clock": "t", "gap_cap": 1}"#,
            "is already a column of the input",
        ),
        (
            r#"{"formulas": [{"name": "a", "tree": ["ewm_mean", ["col", "x"], {"half_life": 1}]}, {"name": "a", "tree": ["ewm_sum", ["col", "x"], {"half_life": 1}]}], "clock": "t", "gap_cap": 1}"#,
            "are named \"a\"",
        ),
        (
            r#"{"formulas": [{"name": "a", "tree": ["ewm_mean", ["col", "nope"], {"half_life": 1}]}], "clock": "t", "gap_cap": 1}"#,
            "the input column \"nope\" is not in the input",
        ),
        (
            r#"{"formulas": [{"name": "a", "tree": ["ewm_mean", ["col", "wide"], {"half_life": 1}]}], "clock": "t", "gap_cap": 1}"#,
            "input must be a number, but it is String",
        ),
        (
            r#"{"formulas": [{"name": "a", "tree": ["-", ["col", "x"], ["lit", 1]]}], "clock": "t", "gap_cap": 1}"#,
            "holds no window operator or increment",
        ),
        (
            r#"{"formulas": [{"name": "a", "tree": ["ewm_mean", ["col", "x"], {"half_life": "1s"}]}], "clock": "t", "gap_cap": 1}"#,
            "half_life is a duration",
        ),
        (
            r#"{"formulas": [{"name": "a", "tree": ["shift", ["col", "x"]]}], "clock": "t", "gap_cap": 1}"#,
            "\"shift\" is not one of the operators a formula can hold",
        ),
    ] {
        let err = match serde_json::from_str::<WindowsConfig>(json) {
            Ok(c) => WindowsRun::new(c, df.schema())
                .err()
                .expect("refused")
                .into_string(),
            Err(e) => e.to_string(),
        };
        assert!(err.contains(says), "{json}: {err}");
    }
}

/// An increment is one row back within the group's session, and seconds
/// on a temporal column.
#[test]
fn an_increment_is_one_row_back_within_the_session() {
    let df = df!(
        "t" => [0.0, 1.0, 2.0, 3.0, 4.0],
        "s" => ["a", "a", "b", "b", "b"],
        "c" => [10.0, 12.0, 15.0, 15.5, 20.0],
    )
    .unwrap();
    let cfg = config(
        r#"{"formulas": [{"name": "dc", "tree": ["increment", ["col", "c"]]},
                {"name": "dt", "tree": ["increment", ["col", "t"]]}],
                "clock": "t", "gap_cap": 100, "session": "s"}"#,
    );
    let mut run = WindowsRun::new(cfg, df.schema()).unwrap();
    let mut out = run.feed(&df.slice(0, 2), None).unwrap();
    out.vstack_mut(&run.feed(&df.slice(2, 3), None).unwrap())
        .unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    let dc: Vec<Option<f64>> = out.column("dc").unwrap().f64().unwrap().iter().collect();
    assert_eq!(dc, vec![None, Some(2.0), None, Some(0.5), Some(4.5)]);
    let dt: Vec<Option<f64>> = out.column("dt").unwrap().f64().unwrap().iter().collect();
    assert_eq!(dt, vec![None, Some(1.0), None, Some(1.0), Some(1.0)]);
}

/// A tick sign is the sign of the increment, the last nonzero
/// one carried over a step of 0, in a slot of its own beside the
/// increments: two tick signs and an increment, interleaved, each read
/// its own input, and the carried signs resume across a save, a step of
/// 0 on each side of the cut. A session change starts them over.
#[test]
fn a_tick_sign_carries_the_last_nonzero_sign_across_a_save() {
    let df = df!(
            "t" => [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "s" => ["a", "a", "a", "a", "a", "a", "b", "b"],
            "p" => [Some(10.0), Some(10.0), Some(11.0), Some(11.0), None, Some(11.0), Some(11.0), Some(9.0)],
            "q" => [5_i64, 4, 4, 4, 6, 6, 6, 6],
        )
        .unwrap();
    let cfg = r#"{"formulas": [{"name": "sp", "tree": ["tick_sign", ["col", "p"]]},
            {"name": "dq", "tree": ["increment", ["col", "q"]]},
            {"name": "sq", "tree": ["tick_sign", ["col", "q"]]}],
            "clock": "t", "gap_cap": 100, "session": "s"}"#;
    let want_p = [
        None,
        None,
        Some(1.0),
        Some(1.0),
        None,
        Some(1.0),
        None,
        Some(-1.0),
    ];
    let want_dq = [
        None,
        Some(-1.0),
        Some(0.0),
        Some(0.0),
        Some(2.0),
        Some(0.0),
        None,
        Some(0.0),
    ];
    let want_q = [
        None,
        Some(-1.0),
        Some(-1.0),
        Some(-1.0),
        Some(1.0),
        Some(1.0),
        None,
        None,
    ];
    let whole = run_whole(cfg, &df);
    assert_eq!(values_of(&whole, "sp"), want_p);
    assert_eq!(values_of(&whole, "dq"), want_dq);
    assert_eq!(values_of(&whole, "sq"), want_q);
    for at in 1..df.height() {
        let mut first = WindowsRun::new(config(cfg), df.schema()).unwrap();
        let mut out = first.feed(&df.slice(0, at), None).unwrap();
        let state = first.save_bytes().unwrap();
        let mut second = WindowsRun::load_bytes(&state, config(cfg), df.schema()).unwrap();
        out.vstack_mut(
            &second
                .feed(&df.slice(at as i64, df.height() - at), None)
                .unwrap(),
        )
        .unwrap();
        out.vstack_mut(&second.finish().unwrap()).unwrap();
        assert!(out.equals_missing(&whole), "cut at {at}");
    }
    // The state holds each tick sign's carried sign past the previous
    // values: three slots, then two signs.
    let mut run = WindowsRun::new(config(cfg), df.schema()).unwrap();
    run.feed(&df.slice(0, 4), None).unwrap();
    assert_eq!(run.incr_state.len(), 1);
    let prev = &run.incr_state[0].prev;
    assert_eq!(prev.len(), 5);
    assert_eq!(&prev[3..], &[1.0, -1.0]);
}

/// The tick rule on one step, from its definition.
#[test]
fn the_tick_rule_on_one_step() {
    for (step, carried, want, then) in [
        (Some(0.5), f64::NAN, Some(1.0), 1.0),
        (Some(-1e-300), 1.0, Some(-1.0), -1.0),
        (Some(0.0), -1.0, Some(-1.0), -1.0),
        (Some(-0.0), 1.0, Some(1.0), 1.0),
        (Some(0.0), f64::NAN, None, f64::NAN),
        (None, 1.0, None, 1.0),
        (None, f64::NAN, None, f64::NAN),
    ] {
        let mut c = carried;
        assert_eq!(tick_sign(step, &mut c), want, "{step:?} {carried}");
        assert_eq!(c.to_bits(), then.to_bits(), "{step:?} {carried}");
    }
}

/// A state resumes only its own call, and goes on from where it was.
#[test]
fn a_state_resumes_its_own_call() {
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut whole = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    let mut want = whole.feed(&df, None).unwrap();
    want.vstack_mut(&whole.finish().unwrap()).unwrap();
    let mut first = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    let mut got = first.feed(&df.slice(0, 3), None).unwrap();
    let bytes = first.save_bytes().unwrap();
    let other = config(
        r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
            {"half_life": 2, "window_size": 2}]}], "clock": "t", "gap_cap": 100}"#,
    );
    let err = WindowsRun::load_bytes(&bytes, other, df.schema())
        .err()
        .expect("refused");
    assert!(err.contains("another call"));
    let mut second = WindowsRun::load_bytes(&bytes, config(FORWARD), df.schema()).unwrap();
    got.vstack_mut(&second.feed(&df.slice(3, 4), None).unwrap())
        .unwrap();
    got.vstack_mut(&second.finish().unwrap()).unwrap();
    assert!(got.equals_missing(&want));
}

/// Review R2, W4, then R4, B1 to B3: a state saved under a slice records
/// the rows of its input the run consumed, skipped or fed, and the
/// input's first clock. A run resumed on the same input skips them, so a
/// chain of sliced runs reads the rest as one run would, rows held from
/// an earlier run and returned counted as none of this input's; a run
/// on an input that starts at another clock skips nothing.
#[test]
fn a_state_saved_under_a_slice_resumes_on_the_same_input() {
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut one = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    let mut whole = one.feed(&df, None).unwrap();
    whole.vstack_mut(&one.finish().unwrap()).unwrap();
    let mut out: Option<DataFrame> = None;
    let mut state: Option<Vec<u8>> = None;
    for _ in 0..4 {
        let mut run = match &state {
            None => WindowsRun::new(config(FORWARD), df.schema()).unwrap(),
            Some(b) => WindowsRun::load_bytes(b, config(FORWARD), df.schema()).unwrap(),
        };
        let part = run.feed(&df, Some(1)).unwrap();
        assert_eq!(part.height(), 1);
        state = Some(run.save_bytes_with(run.consumed(), false).unwrap());
        match &mut out {
            None => out = Some(part),
            Some(o) => {
                o.vstack_mut(&part).unwrap();
            }
        }
    }
    let state = state.unwrap();
    let mut rest = WindowsRun::load_bytes(&state, config(FORWARD), df.schema()).unwrap();
    let mut tail = rest.feed(&df, None).unwrap();
    tail.vstack_mut(&rest.finish().unwrap()).unwrap();
    let mut both = out.unwrap();
    both.vstack_mut(&tail).unwrap();
    assert!(both.equals_missing(&whole), "{both} against {whole}");
    // The next file starts at another clock: none of its rows skipped.
    let later = frame(&[10.0, 11.0, 12.0]);
    let mut next = WindowsRun::load_bytes(&state, config(FORWARD), later.schema()).unwrap();
    let mut got = next.feed(&later, None).unwrap();
    got.vstack_mut(&next.finish().unwrap()).unwrap();
    let fed_later = got
        .column("t")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .flatten()
        .filter(|t| *t >= 10.0)
        .count();
    assert_eq!(fed_later, 3, "{got}");
}

/// Review R2, W5: a state written by version 2 (before round one)
/// loaded with defaults for the fields round one added, and
/// misbehaved; the version is read before the rest, and refused by
/// number.
#[test]
fn a_state_of_another_version_is_refused_by_its_version() {
    #[derive(Serialize)]
    struct Old<'a> {
        magic: &'a str,
        version: u32,
    }
    let input = frame(&[0.0]).schema().clone();
    let old = rmp_serde::to_vec_named(&Old {
        magic: LEGACY_WINDOWS_MAGIC,
        version: 2,
    })
    .unwrap();
    let err = WindowsRun::load_bytes(&old, config(FORWARD), &input)
        .err()
        .expect("refused");
    assert!(
        err.contains(
            "the state's version is 2, but this version of rheon reads versions 9 \
                 to 15"
        ),
        "{err}"
    );
    // A newer enveloped state, and an enveloped one at a version only
    // written before the envelope, are refused by their number.
    let mut run = WindowsRun::new(config(FORWARD), &input).unwrap();
    run.feed(&frame(&[0.0]), None).unwrap();
    let now = run.save_bytes().unwrap();
    for version in [WINDOWS_VERSION + 1, LAST_LEGACY_VERSION] {
        let mut v = rmpv::decode::read_value(&mut &now[..]).unwrap();
        if let rmpv::Value::Map(m) = &mut v {
            m[2].1 = version.into();
        }
        let mut b = Vec::new();
        rmpv::encode::write_value(&mut b, &v).unwrap();
        let err = WindowsRun::load_bytes(&crate::reseal(&b), config(FORWARD), &input)
            .err()
            .expect("refused");
        assert!(
            err.contains(&format!("the state's version is {version}, but")),
            "{err}"
        );
    }
    // Each version written before the envelope loads, and saves the
    // enveloped state.
    for version in [10, 11] {
        let old = crate::envelope::legacy::windows(&now, version);
        let back = WindowsRun::load_bytes(&old, config(FORWARD), &input).unwrap();
        assert_eq!(back.save_bytes().unwrap(), now, "{version}");
    }
    let other = rmp_serde::to_vec_named(&Old {
        magic: "something else",
        version: LAST_LEGACY_VERSION,
    })
    .unwrap();
    let err = WindowsRun::load_bytes(&other, config(FORWARD), &input)
        .err()
        .expect("refused");
    assert!(err.contains("was not saved by a window run"), "{err}");
}

/// Review R5, C1: a resumed run that satisfies its limit from the rows the
/// state held, while a loaded skip still spans the chunks it has fed, saves
/// the rows consumed of the input so far -- the loaded skip included.
#[test]
fn a_resumed_run_fed_one_row_at_a_time_saves_the_whole_count() {
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
    let mut one = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    let mut whole = one.feed(&df, None).unwrap();
    whole.vstack_mut(&one.finish().unwrap()).unwrap();
    // Run 1: head(2) reads rows 0..5: row 0 closes at t = 3, and row 4 at
    // t = 10 closes rows 1..3 at once, so rows 2 and 3 stay held, resolved.
    let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    let mut out = run.feed(&df, Some(2)).unwrap();
    assert_eq!(run.consumed(), 5);
    let state = run.save_bytes_with(run.consumed(), false).unwrap();
    // Run 2, one row a chunk, head(1): the first chunk is skipped, the
    // held row 2 satisfies the limit, the skip still spans four chunks.
    let mut run = WindowsRun::load_bytes(&state, config(FORWARD), df.schema()).unwrap();
    let mut i = 0;
    loop {
        let part = run.feed(&df.slice(i, 1), Some(1)).unwrap();
        i += 1;
        if part.height() == 1 {
            out.vstack_mut(&part).unwrap();
            break;
        }
    }
    assert_eq!(run.consumed(), 5, "the loaded skip not yet applied counts");
    let state = run.save_bytes_with(run.consumed(), false).unwrap();
    // Run 3 on the whole input: the rest, as one run would give it.
    let mut run = WindowsRun::load_bytes(&state, config(FORWARD), df.schema()).unwrap();
    let mut rest = run.feed(&df, None).unwrap();
    rest.vstack_mut(&run.finish().unwrap()).unwrap();
    out.vstack_mut(&rest).unwrap();
    assert!(out.equals_missing(&whole), "{out} against {whole}");
}

/// Review R5, C3/C4 and C6: a sliced state knows its input by the rows it
/// held and its first clock, and refuses another input by name; a state
/// cut short is reported as damaged.
#[test]
fn a_sliced_state_refuses_another_input_and_a_damaged_file_says_so() {
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    run.feed(&df, Some(2)).unwrap();
    let state = run.save_bytes_with(run.consumed(), false).unwrap();
    // The same first clock, other rows after it.
    let other = frame(&[0.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
    let mut run = WindowsRun::load_bytes(&state, config(FORWARD), other.schema()).unwrap();
    let err = run.feed(&other, None).expect_err("refused");
    assert!(err.to_string().contains("another input"), "{err}");
    // The input sliced by hand.
    let mut run = WindowsRun::load_bytes(&state, config(FORWARD), df.schema()).unwrap();
    let err = run.feed(&df.slice(2, 5), None).expect_err("refused");
    assert!(err.to_string().contains("another input"), "{err}");
    // A file cut short.
    let cut = &state[..state.len() / 2];
    let err = WindowsRun::load_bytes(cut, config(FORWARD), df.schema())
        .err()
        .expect("refused");
    assert!(err.contains("damaged"), "{err}");
}

/// Review R4, A2 and R6, D5: a windows state version moves the bank's
/// schema with it, since a bank file embeds one per formula target and
/// refuses an older form by number, not at a group's first chunk.
#[test]
fn a_windows_state_version_moves_the_banks_schema_with_it() {
    // 28 to 39 moved for `ew_cov`'s PCA cadence,
    // `micro`'s pruning, the models' window cadence, `lasso`'s
    // per-target thresholds, the windows' stamps, the quantile fit's
    // band systems, the embargo's elapsed clock, `coef_every` on the
    // clock, `bocpd`'s hazard on the clock, the solve, component and
    // checkpoint cadences on the exact clock, `ewridge`'s kept
    // systems, the stream's and the bank's state and task 195's
    // residual scales, scaler, warm-up and per-target thresholds (tasks
    // 161, 163, 162, 174, 175, 170, 176, 178, 179, 180, 186, 194, 195),
    // and 41 for the models' window edge and task 196's names, the
    // windows state unchanged: a schema may move alone, a windows
    // version may not. 42 (task 200) moved with windows state 8, an
    // integer clock held as an integer in its rows; 43 for the relative
    // targets' removal (task 201), 44 for task 202's target spreads,
    // 45 for task 116's readiness statistics, 46 for the readiness
    // notices' waits (task 208) and 47 for task 206's warm-up of a
    // standardizing fit, the windows state unchanged; 48 (task 212)
    // with windows state 9, a variance's queue six wide; 49 for task
    // 211's `kalman` anchor and clocks, 50 for task 214's `kalman`
    // prior from a median, 51 for task 215's `ftrl` without its
    // penalty scale and the clusters' centres as pairs, 52 for task
    // 217's row counts in `ewridge`'s and `lasso`'s cross-moments, and
    // 54 for task 223 (b)'s `audit`, 55 for task 221's diagnostics
    // with a memory of their own, 56 for its parts (d)-(h), 57 for
    // review round 6's tails and scored inflation, 59 for task 233's
    // smallest change in an audit, 60 for task 232's Newey-West sums
    // and 61 for review round 7's residual lag products on a quantile
    // fit (A-2), task 244's `bias` in an `ew_cov` and its 1.0 names, and
    // 62 for task 257's held targets of a blocked `ewridge`, the first
    // change since 1.0, whose loader reads 61, and 63 for task 271's
    // Huber scale, whose loader reads 62: the windows state unchanged.
    // Windows state 10 (task 272 (d), a quantile's sketches) within
    // schema 63: this build reads a 9 core as a 10 without a quantile,
    // so a 62 or 63 bank holding window targets resumes here, and no
    // released build reads a 63 bank (1.2.0 refuses it by number), so
    // none meets a 10 core at a group's first chunk. 64 for task 277's
    // refit of `huber`'s and `sgd`'s first rows, whose loader reads 63:
    // the windows state unchanged. 65 for task 323's MAD read exactly
    // from a refit's held `|e|`, whose loader reads 64: unchanged too.
    // 66 for task 327's diagonal `hmm` and `ew_class` states and task
    // 328's count of the rows `corrchange`'s window ring has taken,
    // whose loaders read 65: unchanged too. 67 for task 340's pending
    // notices saved with a stream and `marginal`'s mark cleared at its
    // freeze, whose loaders read 66: unchanged too. 68 for task 281
    // (3)'s square root of a `kalman`'s covariance, whose loader reads
    // 67: unchanged too. 69 for task 479's factor form of a wide
    // `full` `hmm` state, whose loader reads 68: unchanged too. 70 for
    // `hmm`'s stabilized prior, whose loader reads 69: unchanged too.
    // 71 with windows state 11 (review P7): both end with a checksum;
    // a 70 bank's 10 cores have none and load as they did. 72 with
    // windows state 12 (PLAN 542): both written in the envelope every
    // state file shares; a 71 bank's 11 cores load as they did.
    // Windows state 13 (task 561, the extrema) and 14 (task 564, the
    // run lengths and clock drifts) within schema 72, which no release
    // has written: this build reads a 12 core as a 14 without an
    // extremum or a row operator's state, and a 13 as a 14 without the
    // latter, so a 72 bank holding window targets resumes here, and a
    // released build refuses a 72 bank by number.
    // Windows state 14 also holds PLAN 563's invert node on the same
    // reading: a 13 core loads as a 14 without an invert. Windows
    // state 14 holds PLAN 490's STL rings likewise: a 13 core loads as a 14
    // without a ring.
    // Windows state 15 (task 78, placeholder sum kernel removal) within
    // schema 72, which no release has written: a row-only call now carries
    // an empty clock-only core (0 kernels, 0 operators) instead of the
    // 1-kernel, 1-sum stand-in that versions 9 to 14 wrote; this build
    // strips that stand-in when loading a 9..=14 row-only state and loads
    // every other 9..=14 state unchanged, while a released build refuses a
    // 72 bank by number.
    assert_eq!((WINDOWS_VERSION, rheon_core::SCHEMA_VERSION), (15, 72));
    assert_eq!(MIN_WINDOWS_VERSION, 9);
}

/// Review R6, D2: a run on the next file under a slice keeps the first
/// file's unresolved rows held ahead of its own -- rows go out in order,
/// so none of its own went out while one of theirs waited -- and the
/// state then holds more rows than it consumed of this input. The
/// identity is the last `consumed` of the held rows, this input's own;
/// a resume on the second file goes on.
#[test]
fn a_sliced_run_on_the_next_file_resumes_with_the_first_files_rows_held() {
    let first = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let second = frame(&[10.0, 11.0, 12.0]);
    let whole = first.vstack(&second).unwrap();
    let mut one = WindowsRun::new(config(FORWARD), whole.schema()).unwrap();
    let mut want = one.feed(&whole, None).unwrap();
    want.vstack_mut(&one.finish().unwrap()).unwrap();
    // The first file, unsliced: rows 3..6 stay held, their windows open.
    let mut run = WindowsRun::new(config(FORWARD), first.schema()).unwrap();
    let mut out = run.feed(&first, None).unwrap();
    assert_eq!(run.held(), 3);
    let state = run.save_bytes().unwrap();
    // The second file under head(1): its first row closes the three, one
    // goes out, two stay held ahead of it; one row of it consumed.
    let mut run = WindowsRun::load_bytes(&state, config(FORWARD), second.schema()).unwrap();
    out.vstack_mut(&run.feed(&second, Some(1)).unwrap())
        .unwrap();
    assert_eq!((run.consumed(), run.held()), (1, 3));
    let state = run.save_bytes_with(run.consumed(), false).unwrap();
    // The second file resumed, unsliced: the rest, as one run would give it.
    let mut run = WindowsRun::load_bytes(&state, config(FORWARD), second.schema()).unwrap();
    out.vstack_mut(&run.feed(&second, None).unwrap()).unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    assert!(out.equals_missing(&want), "{out} against {want}");
}

/// Review R6, D1: a backward operator under "left" resolves a row at its
/// own push, so a sliced run holds no row, and the held rows identify
/// nothing; the state keeps the last row it read and refuses another
/// input by it, with a clock column and without (a row-count clock
/// repeats no stamp, so every operator resolves at the push).
#[test]
fn a_sliced_state_that_holds_no_rows_knows_its_input_by_its_last_row() {
    const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
    const BARE: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1}]}]}"#;
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let other = frame(&[0.0, 1.5, 2.5, 3.5, 4.5]);
    for (cfg, other) in [(LEFT, other), (BARE, df.slice(3, 4))] {
        let mut one = WindowsRun::new(config(cfg), df.schema()).unwrap();
        let mut want = one.feed(&df, None).unwrap();
        want.vstack_mut(&one.finish().unwrap()).unwrap();
        let mut run = WindowsRun::new(config(cfg), df.schema()).unwrap();
        let mut out = run.feed(&df, Some(3)).unwrap();
        assert_eq!(
            (out.height(), run.consumed(), run.held()),
            (3, 3, 0),
            "{cfg}"
        );
        let state = run.save_bytes_with(run.consumed(), false).unwrap();
        let mut run = WindowsRun::load_bytes(&state, config(cfg), df.schema()).unwrap();
        let err = run.feed(&other, None).expect_err("refused");
        assert!(err.to_string().contains("another input"), "{cfg}: {err}");
        let mut run = WindowsRun::load_bytes(&state, config(cfg), df.schema()).unwrap();
        out.vstack_mut(&run.feed(&df, None).unwrap()).unwrap();
        out.vstack_mut(&run.finish().unwrap()).unwrap();
        assert!(out.equals_missing(&want), "{cfg}: {out} against {want}");
    }
}

/// Review R6, D3 and D4: an input that starts elsewhere is the next file
/// when the clock policy takes its first row as a step forward or a new
/// start (a step back past `restart_after_step_back`, a new session),
/// and is refused when it starts at the last stamp the state read (a
/// file boundary inside a tied stamp cannot be told from the same input
/// sliced inside it) or steps back where the policy refuses it.
#[test]
fn a_sliced_state_takes_a_new_start_by_the_policys_word_for_the_next_file() {
    let day = |t: &[f64], s: &str| {
        df!(
            "t" => t,
            "x" => t.iter().map(|v| v * 10.0).collect::<Vec<_>>(),
            "s" => vec![s; t.len()],
        )
        .unwrap()
    };
    let day1 = day(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "a");
    let day2 = day(&[0.5, 1.5, 2.5, 3.5], "b");
    let restart = r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
            {"half_life": 1, "window_size": 2}]}], "clock": "t", "gap_cap": 100,
            "restart_after_step_back": 2}"#;
    let session = r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
            {"half_life": 1, "window_size": 2}]}], "clock": "t", "gap_cap": 100,
            "session": "s"}"#;
    for cfg in [restart, session] {
        let whole = day1.vstack(&day2).unwrap();
        let mut one = WindowsRun::new(config(cfg), whole.schema()).unwrap();
        let mut want = one.feed(&whole, None).unwrap();
        want.vstack_mut(&one.finish().unwrap()).unwrap();
        // Day 1 under a slice its input exhausts: a state with a skip.
        let mut run = WindowsRun::new(config(cfg), day1.schema()).unwrap();
        let mut out = run.feed(&day1, Some(100)).unwrap();
        assert!(run.held() > 0 && run.consumed() == 7);
        let state = run.save_bytes_with(run.consumed(), true).unwrap();
        // Day 2 steps back: a new start by the policy's word, the next file.
        let mut run = WindowsRun::load_bytes(&state, config(cfg), day2.schema()).unwrap();
        out.vstack_mut(&run.feed(&day2, None).unwrap()).unwrap();
        out.vstack_mut(&run.finish().unwrap()).unwrap();
        assert!(out.equals_missing(&want), "{cfg}: {out} against {want}");
    }
    // Review R7, E2: day 2 at the same first stamp as day 1. With a
    // session column the first row's session tells the file from the
    // saved input, and the policy takes it; without one it is the saved
    // input until the rows differ, refused by name.
    let same = day(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "b");
    let whole = day1.vstack(&same).unwrap();
    let mut one = WindowsRun::new(config(session), whole.schema()).unwrap();
    let mut want = one.feed(&whole, None).unwrap();
    want.vstack_mut(&one.finish().unwrap()).unwrap();
    let mut run = WindowsRun::new(config(session), day1.schema()).unwrap();
    let mut out = run.feed(&day1, Some(100)).unwrap();
    let state = run.save_bytes_with(run.consumed(), true).unwrap();
    let mut run = WindowsRun::load_bytes(&state, config(session), same.schema()).unwrap();
    out.vstack_mut(&run.feed(&same, None).unwrap()).unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    assert!(out.equals_missing(&want), "{out} against {want}");
    let mut run = WindowsRun::new(config(restart), day1.schema()).unwrap();
    run.feed(&day1, Some(100)).unwrap();
    let state = run.save_bytes_with(run.consumed(), true).unwrap();
    let mut run = WindowsRun::load_bytes(&state, config(restart), same.schema()).unwrap();
    let err = run.feed(&same, None).expect_err("refused");
    assert!(err.to_string().contains("another input"), "{err}");
    // Under the default policy a step back is refused, and so is a tie.
    let plain = r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
            {"half_life": 1, "window_size": 2}]}], "clock": "t", "gap_cap": 100}"#;
    let mut run = WindowsRun::new(config(plain), day1.schema()).unwrap();
    run.feed(&day1, Some(100)).unwrap();
    let state = run.save_bytes_with(run.consumed(), true).unwrap();
    let mut run = WindowsRun::load_bytes(&state, config(plain), day2.schema()).unwrap();
    let err = run.feed(&day2, None).expect_err("refused");
    assert!(
        err.to_string()
            .contains("before the last row the state read"),
        "{err}"
    );
    let tied = day(&[6.0, 7.0, 8.0], "a");
    let mut run = WindowsRun::load_bytes(&state, config(plain), tied.schema()).unwrap();
    let err = run.feed(&tied, None).expect_err("refused");
    assert!(
        err.to_string()
            .contains("at the last clock value the state read"),
        "{err}"
    );
}

/// Review R5, C1 (the plan's row, unpinned) and C3: a resumed run that
/// saw no row saves the identity it loaded, so the chain goes on from
/// that state as from the one before it; a save that says the input
/// ended refuses a skip still pending, one that does not takes it.
#[test]
fn a_resumed_run_that_saw_no_row_saves_the_identity_it_loaded() {
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
    let mut one = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    let mut whole = one.feed(&df, None).unwrap();
    whole.vstack_mut(&one.finish().unwrap()).unwrap();
    let mut run = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    let mut out = run.feed(&df, Some(2)).unwrap();
    assert_eq!(run.consumed(), 5);
    let first = run.save_bytes_with(run.consumed(), false).unwrap();
    // Loaded and saved again without a row: the same identity and count.
    let run = WindowsRun::load_bytes(&first, config(FORWARD), df.schema()).unwrap();
    assert_eq!(run.consumed(), 5);
    let again = run.save_bytes_with(run.consumed(), false).unwrap();
    let mut run = WindowsRun::load_bytes(&again, config(FORWARD), df.schema()).unwrap();
    out.vstack_mut(&run.feed(&df, None).unwrap()).unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    assert!(out.equals_missing(&whole), "{out} against {whole}");
    // Mid-skip: the input is not over unless the caller says so.
    let mut run = WindowsRun::load_bytes(&first, config(FORWARD), df.schema()).unwrap();
    run.feed(&df.slice(0, 2), None).unwrap();
    assert_eq!(run.consumed(), 5);
    run.save_bytes_with(run.consumed(), false)
        .expect("a slice satisfied mid-skip");
    let err = run
        .save_bytes_with(run.consumed(), true)
        .expect_err("refused");
    assert!(err.contains("ended after 2"), "{err}");
    // Review R7, E5: the identity itself survives a re-save without a
    // row -- the last row read, where nothing is held, refuses another
    // input with the same first clock.
    const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
    let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
    run.feed(&df, Some(3)).unwrap();
    let first = run.save_bytes_with(run.consumed(), false).unwrap();
    let run = WindowsRun::load_bytes(&first, config(LEFT), df.schema()).unwrap();
    let again = run.save_bytes_with(run.consumed(), false).unwrap();
    let other = frame(&[0.0, 1.5, 2.5, 3.5, 4.5]);
    let mut run = WindowsRun::load_bytes(&again, config(LEFT), other.schema()).unwrap();
    let err = run.feed(&other, None).expect_err("refused");
    assert!(err.to_string().contains("another input"), "{err}");
}

/// Review R7, E3: the last row read is a sliced state's identity and
/// nothing else's, so an unsliced state that holds no rows resumes on an
/// input with a column more, as it did before the row was kept; a
/// sliced one names the columns it has against the frame's.
#[test]
fn the_last_row_binds_a_sliced_state_only() {
    const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
    let df = frame(&[0.0, 1.0, 2.0, 3.0]);
    let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
    run.feed(&df, None).unwrap();
    assert_eq!(run.held(), 0);
    let state = run.save_bytes().unwrap();
    // The next file, with a column more.
    let wider = frame(&[4.0, 5.0, 6.0, 7.0])
        .lazy()
        .with_columns([lit(1.0).alias("extra")])
        .collect()
        .unwrap();
    let mut run = WindowsRun::load_bytes(&state, config(LEFT), wider.schema()).unwrap();
    let out = run.feed(&wider, None).unwrap();
    assert_eq!(out.height(), 4);
    assert!(out.get_column_names().iter().any(|c| c.as_str() == "extra"));
    let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
    run.feed(&df, Some(2)).unwrap();
    let state = run.save_bytes_with(run.consumed(), false).unwrap();
    let err = WindowsRun::load_bytes(&state, config(LEFT), wider.schema())
        .err()
        .expect("refused");
    assert!(
        err.contains("the last row the state read has columns"),
        "{err}"
    );
}

/// Task 158, E13: a state saved without a slice refuses an input that
/// starts at the last stamp it read -- here one that repeats its last
/// row, which came out twice -- as a sliced state does, and goes on with
/// one that starts after it.
#[test]
fn an_unsliced_state_refuses_an_input_at_the_last_stamp_it_read() {
    const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
    let df = frame(&[0.0, 1.0, 2.0, 3.0]);
    let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
    run.feed(&df, None).unwrap();
    let state = run.save_bytes().unwrap();
    let again = frame(&[3.0, 4.0, 5.0]);
    let mut run = WindowsRun::load_bytes(&state, config(LEFT), again.schema()).unwrap();
    let err = run.feed(&again, None).expect_err("refused");
    assert!(
        err.to_string()
            .contains("at the last clock value the state read"),
        "{err}"
    );
    let next = frame(&[4.0, 5.0, 6.0]);
    let mut run = WindowsRun::load_bytes(&state, config(LEFT), next.schema()).unwrap();
    assert_eq!(run.feed(&next, None).unwrap().height(), 3);
    // A fresh run has read no stamp.
    let mut run = WindowsRun::new(config(LEFT), again.schema()).unwrap();
    assert_eq!(run.feed(&again, None).unwrap().height(), 3);
}

/// PLAN 567: `settle` reads the backward windows waiting on the last
/// stamp as they stand, and the rows ready go out; saved after it, the
/// state continues to the bit at every split point, with the first run's
/// rows, then the second's, one run's rows exactly. A forward window
/// beside them keeps the rows it still holds.
#[test]
fn a_settled_state_continues_to_the_bit_at_every_split() {
    const MIXED: &str = r#"{"formulas": [
            {"name": "b", "tree": ["ewm_mean", ["col", "x"], {"half_life": 2, "window_size": 3}]},
            {"name": "l", "tree": ["ewm_sum", ["col", "x"], {"half_life": 2, "closed": "left"}]},
            {"name": "a", "tree": ["rewm_sum", ["col", "x"], {"half_life": 2, "window_size": 2}]}],
            "clock": "t", "gap_cap": 100}"#;
    let stamps = [0.0, 1.0, 2.0, 2.0, 3.0, 5.0, 6.0, 6.0, 7.0, 9.0];
    let df = frame(&stamps);
    let mut whole = WindowsRun::new(config(MIXED), df.schema()).unwrap();
    let mut one = whole.feed(&df, None).unwrap();
    one.vstack_mut(&whole.finish().unwrap()).unwrap();
    for k in 1..stamps.len() {
        if stamps[k] == stamps[k - 1] {
            continue;
        }
        let head = df.slice(0, k);
        let mut run = WindowsRun::new(config(MIXED), df.schema()).unwrap();
        let mut got = run.feed(&head, None).unwrap();
        let settled = run.settle().unwrap();
        // The backward windows of every row so far have their values: a
        // row is held only for the window looking ahead.
        let (b, l) = (settled.column("b").unwrap(), settled.column("l").unwrap());
        assert_eq!(b.null_count() + l.null_count(), 0, "k={k}");
        got.vstack_mut(&settled).unwrap();
        let state = run.save_bytes().unwrap();
        let rest = df.slice(k as i64, df.height() - k);
        let mut run = WindowsRun::load_bytes(&state, config(MIXED), rest.schema()).unwrap();
        assert_eq!(run.save_bytes().unwrap(), state, "k={k}: loaded, saved");
        got.vstack_mut(&run.feed(&rest, None).unwrap()).unwrap();
        got.vstack_mut(&run.finish().unwrap()).unwrap();
        assert_eq!(got.height(), one.height(), "k={k}");
        for name in ["b", "l", "a"] {
            let (g, w) = (got.column(name).unwrap(), one.column(name).unwrap());
            let bits = |c: &Column| -> Vec<Option<u64>> {
                c.f64()
                    .unwrap()
                    .iter()
                    .map(|v| v.map(f64::to_bits))
                    .collect()
            };
            assert_eq!(bits(g), bits(w), "k={k} {name}");
        }
    }
}

/// Task 159 (F1): a run marked itself started, and kept its input's
/// first row, before the identity verdict, so a refused first chunk was
/// taken whole on the next call. A refused first chunk leaves the run as
/// it was: fed again, it is refused again, with the same words.
#[test]
fn a_refused_first_chunk_is_refused_again() {
    const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
    let df = frame(&[0.0, 1.0, 2.0, 3.0]);
    let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
    run.feed(&df, None).unwrap();
    let state = run.save_bytes().unwrap();
    let again = frame(&[3.0, 4.0, 5.0]);
    let mut run = WindowsRun::load_bytes(&state, config(LEFT), again.schema()).unwrap();
    let first = run.feed(&again, None).expect_err("refused").to_string();
    let second = run
        .feed(&again, None)
        .expect_err("refused again")
        .to_string();
    assert!(
        first.contains("at the last clock value the state read"),
        "{first}"
    );
    assert_eq!(first, second);
    // And the run goes on with the next file as if nothing had been fed.
    let next = frame(&[4.0, 5.0, 6.0]);
    assert_eq!(run.feed(&next, None).unwrap().height(), 3);
}

/// Task 159 (W1): a restart on the stream's clock starts every group's
/// clock over with its windows, so a group's next row within
/// `restart_after_step_back` of its last is the first on a fresh clock,
/// not a late row: C at 52, 8 after C at 60, once B at 50 restarted the
/// stream after A at 100.
#[test]
fn a_stream_restart_restarts_every_groups_clock() {
    const GROUPED: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_sum", ["col", "x"],
            {"half_life": 1e300}]}], "clock": "t", "gap_cap": 1e9, "group": "g",
            "restart_after_step_back": 10}"#;
    let df = df!(
        "t" => [0.0, 60.0, 100.0, 50.0, 52.0],
        "g" => ["A", "C", "A", "B", "C"],
        "x" => [1.0; 5]
    )
    .unwrap();
    let mut run = WindowsRun::new(config(GROUPED), df.schema()).unwrap();
    let fed = run.feed(&df, None).expect("every row taken").height();
    let rest = run.finish().unwrap().height();
    assert_eq!(fed + rest, 5);
}

/// Task 159 (F2): a key is its value's text, so a group column of another
/// dtype on a resumed input started every key over, silently. The state
/// carries the group and session columns' dtypes, and refuses another.
#[test]
fn a_state_refuses_a_group_column_of_another_dtype() {
    const GROUPED: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1}]}], "clock": "t", "gap_cap": 100, "group": "g"}"#;
    let df = df!("t" => [0.0, 1.0, 2.0, 3.0], "x" => [1.0, 2.0, 3.0, 4.0], "g" => [1i64, 1, 2, 2])
        .unwrap();
    let mut run = WindowsRun::new(config(GROUPED), df.schema()).unwrap();
    run.feed(&df, None).unwrap();
    let state = run.save_bytes().unwrap();
    let next = df!("t" => [4.0, 5.0], "x" => [5.0, 6.0], "g" => [1.0f64, 2.0]).unwrap();
    let err = WindowsRun::load_bytes(&state, config(GROUPED), next.schema())
        .err()
        .expect("refused");
    assert!(
        err.contains("group column \"g\": i64") && err.contains("f64"),
        "{err}"
    );
    let same = df!("t" => [4.0, 5.0], "x" => [5.0, 6.0], "g" => [1i64, 2]).unwrap();
    WindowsRun::load_bytes(&state, config(GROUPED), same.schema()).unwrap();
}

/// Review R8, F1: on a row-count clock every row is a step forward, so
/// a step forward says nothing; without a clock column the next file
/// begins with a new session, and a hand slice starting in the session
/// the state last read is refused.
#[test]
fn without_a_clock_the_next_file_begins_with_a_new_session() {
    const ROWS: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1}]}], "session": "s"}"#;
    let rows = |x: &[f64], s: &[&str]| df!("x" => x, "s" => s).unwrap();
    let df = rows(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &["a", "a", "a", "b", "b", "b"],
    );
    let mut run = WindowsRun::new(config(ROWS), df.schema()).unwrap();
    let mut out = run.feed(&df, Some(4)).unwrap();
    assert_eq!((out.height(), run.consumed(), run.held()), (4, 4, 0));
    let state = run.save_bytes_with(run.consumed(), false).unwrap();
    for cut in [3, 5] {
        let mut run = WindowsRun::load_bytes(&state, config(ROWS), df.schema()).unwrap();
        let err = run
            .feed(&df.slice(cut, df.height() - cut as usize), None)
            .expect_err("refused");
        assert!(
            err.to_string().contains("continues the session last read"),
            "{cut}: {err}"
        );
    }
    let next = rows(&[7.0, 8.0], &["c", "c"]);
    let whole = df.head(Some(4)).vstack(&next).unwrap();
    let mut one = WindowsRun::new(config(ROWS), whole.schema()).unwrap();
    let mut want = one.feed(&whole, None).unwrap();
    want.vstack_mut(&one.finish().unwrap()).unwrap();
    let mut run = WindowsRun::load_bytes(&state, config(ROWS), next.schema()).unwrap();
    out.vstack_mut(&run.feed(&next, None).unwrap()).unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    assert!(out.equals_missing(&want), "{out} against {want}");
}

/// Review R8, F2: where the policy refuses the first row of an input
/// that starts elsewhere, the refusal says what the policy said -- under
/// `group` a next-day file's step back is refused on the stream's clock
/// whatever its session, as the unsliced path refuses it, naming
/// `restart_after_step_back`.
#[test]
fn a_refused_first_row_carries_the_policys_words() {
    const GROUPED: &str = r#"{"formulas": [{"name": "f", "tree": ["rewm_mean", ["col", "x"],
            {"half_life": 1, "window_size": 2}]}], "clock": "t", "gap_cap": 100,
            "group": "g", "session": "s"}"#;
    let day = |t: &[f64], s: &str| {
        df!(
            "t" => t,
            "x" => t.iter().map(|v| v * 10.0).collect::<Vec<_>>(),
            "g" => vec!["k"; t.len()],
            "s" => vec![s; t.len()],
        )
        .unwrap()
    };
    let day1 = day(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "a");
    let day2 = day(&[0.0, 1.0, 2.0, 3.0], "b");
    let mut run = WindowsRun::new(config(GROUPED), day1.schema()).unwrap();
    run.feed(&day1, Some(100)).unwrap();
    let state = run.save_bytes_with(run.consumed(), true).unwrap();
    let mut run = WindowsRun::load_bytes(&state, config(GROUPED), day2.schema()).unwrap();
    let err = run.feed(&day2, None).expect_err("refused").to_string();
    assert!(
        err.contains("starts before the last row the state read")
            && err.contains("restart_after_step_back"),
        "{err}"
    );
}

/// Review R7, E4 and R8, F3: a run that stops inside the skip (a limit
/// of 0 on a chunk shorter than the skip) saves the count and the
/// identity it loaded, not the partial skip's last row, so the next
/// resume on the same input goes on.
#[test]
fn a_save_inside_a_pending_skip_keeps_the_loaded_identity() {
    const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut one = WindowsRun::new(config(LEFT), df.schema()).unwrap();
    let mut whole = one.feed(&df, None).unwrap();
    whole.vstack_mut(&one.finish().unwrap()).unwrap();
    let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
    let mut out = run.feed(&df, Some(3)).unwrap();
    let state = run.save_bytes_with(run.consumed(), false).unwrap();
    let mut run = WindowsRun::load_bytes(&state, config(LEFT), df.schema()).unwrap();
    assert_eq!(run.feed(&df.slice(0, 1), Some(0)).unwrap().height(), 0);
    assert_eq!(run.consumed(), 3);
    let state = run.save_bytes_with(run.consumed(), false).unwrap();
    let mut run = WindowsRun::load_bytes(&state, config(LEFT), df.schema()).unwrap();
    out.vstack_mut(&run.feed(&df, None).unwrap()).unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    assert!(out.equals_missing(&whole), "{out} against {whole}");
}

/// Review R8, F5: a state saved under a slice carries the rows it knows
/// its input by; one that carries none is damaged, refused at load.
#[test]
fn a_sliced_state_without_its_identity_is_damaged() {
    const LEFT: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {"half_life": 1, "closed": "left"}]}], "clock": "t", "gap_cap": 100}"#;
    let df = frame(&[0.0, 1.0, 2.0, 3.0]);
    let mut run = WindowsRun::new(config(LEFT), df.schema()).unwrap();
    run.feed(&df, Some(2)).unwrap();
    let state = run.save_bytes_with(run.consumed(), false).unwrap();
    let file: FileIn = rmp_serde::from_slice(&state).unwrap();
    let stripped = rmp_serde::to_vec_named(&FileOut {
        magic: WINDOWS_MAGIC,
        kind: WINDOWS_KIND,
        version: WINDOWS_VERSION,
        config: &file.config,
        core: &file.core,
        held: &file.held,
        increments: &file.increments,
        stream_clock: file.stream_clock,
        key_dtypes: &file.key_dtypes,
        resume_skip: file.resume_skip,
        resume_first: file.resume_first,
        resume_first_session: file.resume_first_session,
        last_row: &[],
        checksum: None,
    })
    .map(|b| crate::reseal(&b))
    .unwrap();
    let err = WindowsRun::load_bytes(&stripped, config(LEFT), df.schema())
        .err()
        .expect("refused");
    assert!(err.contains("damaged"), "{err}");
}

/// Review 2026-10-06, PD1: the core a state carries is held to its own
/// invariants at load, beyond the kernels, the operators, the clock and
/// the held count the loader compared: a held value short, an output
/// count below the operators, a queue of another width, a silent-list
/// link past the groups, more rows ready than held, or a waiting row
/// before the held ones loaded, and `finish`'s `emit` or the next
/// `push` indexed `held_values[r * n_out + o]` past its end.
#[test]
fn a_state_whose_core_is_damaged_is_refused() {
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut first = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    first.feed(&df.slice(0, 4), None).unwrap();
    let bytes = first.save_bytes().unwrap();
    let file: FileIn = rmp_serde::from_slice(&bytes).unwrap();
    assert!(file.core.held() > 0, "rows wait for their forward windows");
    for what in [
        "held_values",
        "n_outputs",
        "queue",
        "link",
        "ready",
        "waiting",
    ] {
        let mut core = file.core.clone();
        core.damage(what);
        let damaged = rmp_serde::to_vec_named(&FileOut {
            magic: WINDOWS_MAGIC,
            kind: WINDOWS_KIND,
            version: WINDOWS_VERSION,
            config: &file.config,
            core: &core,
            held: &file.held,
            increments: &file.increments,
            stream_clock: file.stream_clock,
            key_dtypes: &file.key_dtypes,
            resume_skip: file.resume_skip,
            resume_first: file.resume_first,
            resume_first_session: file.resume_first_session,
            last_row: &file.last_row,
            checksum: None,
        })
        .map(|b| crate::reseal(&b))
        .unwrap();
        match WindowsRun::load_bytes(&damaged, config(FORWARD), df.schema()) {
            Err(err) => assert!(err.contains("the state is damaged"), "{what}: {err}"),
            Ok(mut run) => {
                let fed = run.feed(&df.slice(4, 2), None).map(|f| f.height());
                let done = run.finish().map(|f| f.height());
                panic!("{what}: loaded, fed {fed:?} and finished {done:?}");
            }
        }
    }
    assert!(WindowsRun::load_bytes(&bytes, config(FORWARD), df.schema()).is_ok());
}

/// A run over the whole of `df` in one chunk, then the end of input.
fn run_whole(cfg: &str, df: &DataFrame) -> DataFrame {
    let mut run = WindowsRun::new(config(cfg), df.schema()).unwrap();
    let mut out = run.feed(df, None).unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    out
}

fn values_of(df: &DataFrame, c: &str) -> Vec<Option<f64>> {
    df.column(c).unwrap().f64().unwrap().iter().collect()
}

/// A formula whose value is its row's increment of `input`: a sum over
/// `(t - w, t]` holds only the row itself when stamps are at least `w`
/// apart, at weight `2^0 = 1`, and is null where the increment is.
fn increment_formula(input: &str, half_life: &str, window: &str) -> String {
    format!(
        r#"{{"name": "d", "tree": ["ewm_sum", ["increment", {input}],
            {{"half_life": {half_life}, "window_size": {window}, "min_samples": 1}}]}}"#
    )
}

/// Task 252: the refusals of a call's plan no other test reached, each
/// by its words: an output named nothing or under a reserved prefix, an
/// increment of an increment, an increment or an operator whose input
/// Polars cannot type or that is text, a clock or group column the
/// frame has not got, and a `like=` spec reading a column it has not.
#[test]
fn a_call_the_plan_cannot_run_is_refused_by_name() {
    let df = df!(
        "t" => [0.0, 1.0],
        "x" => [1.0, 2.0],
        "s" => ["a", "b"],
    )
    .unwrap();
    let one = |name: &str, tree: &str, policy: &str| {
        format!(r#"{{"formulas": [{{"name": "{name}", "tree": {tree}}}]{policy}}}"#)
    };
    let clocked = r#", "clock": "t", "gap_cap": 10"#;
    let sum = r#"["ewm_sum", ["col", "x"], {"half_life": 1}]"#;
    for (json, says) in [
        (one("", sum, clocked), "an output cannot be named \"\""),
        (
            one("@po:y", sum, clocked),
            "an output cannot be named \"@po:y\"",
        ),
        (
            one("@in:0", sum, clocked),
            "an output cannot be named \"@in:0\"",
        ),
        (
            one(
                "d",
                r#"["ewm_sum", ["increment", ["increment", ["col", "x"]]], {"half_life": 1}]"#,
                clocked,
            ),
            "d: an increment's input holds another increment",
        ),
        (
            one("d", r#"["increment", ["col", "s"]]"#, clocked),
            "d: an increment's input must be a number or a temporal column, but it is String",
        ),
        (
            one(
                "d",
                r#"["increment", ["-", ["col", "s"], ["lit", 1]]]"#,
                clocked,
            ),
            "d: Polars cannot compute the increment's input (",
        ),
        (
            one(
                "m",
                r#"["ewm_sum", ["-", ["col", "s"], ["lit", 1]], {"half_life": 1}]"#,
                clocked,
            ),
            "m: Polars cannot compute ewm_sum's input (",
        ),
        (
            one("m", sum, r#", "clock": "nope", "gap_cap": 10"#),
            "with_windows: the clock column \"nope\" is not in the input, whose columns are \
                 [\"t\", \"x\", \"s\"]. Name a column the input has, or add \"nope\" to it.",
        ),
        (
            one("m", sum, r#", "group": "g""#),
            "with_windows: the group column \"g\" is not in the input",
        ),
        (
            one(
                "m",
                sum,
                r#", "clock": "t", "gap_cap": 10, "like": {"spec": "fit", "accept": ["x", "w"]}"#,
            ),
            "with_windows: the spec \"fit\" given as like= reads column \"w\", which is not in \
                 the input",
        ),
    ] {
        let err = WindowsRun::new(config(&json), df.schema())
            .err()
            .unwrap_or_else(|| panic!("{json}: ran"));
        assert!(err.contains(says), "{json}: {err}");
    }
}

/// Task 252: on one clock every clock parameter is written the same
/// way, as a spec's are: a duration needs a temporal clock column,
/// named or not, and a number needs a numeric one; and on a `Date`
/// clock, whose smallest step is a day, a cap or a restart below a day
/// is refused, saying why.
#[test]
fn durations_and_numbers_must_match_the_clock() {
    let day = |d: &[i32]| {
        Series::new("d".into(), d)
            .cast(&DataType::Date)
            .unwrap()
            .into_column()
    };
    let ms = Series::new("t".into(), [0i64, 1_000])
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        .unwrap()
        .into_column();
    let df = DataFrame::new(
        2,
        vec![
            ms,
            day(&[19_723, 19_724]),
            Column::new("f".into(), [0.0, 1.0]),
            Column::new("x".into(), [1.0, 2.0]),
        ],
    )
    .unwrap();
    let call = |half_life: &str, policy: &str| {
        format!(
            r#"{{"formulas": [{{"name": "m", "tree": ["ewm_sum", ["col", "x"],
                {{"half_life": {half_life}}}]}}]{policy}}}"#
        )
    };
    for (json, says) in [
        (
            call(r#""1s""#, r#", "clock": "f", "gap_cap": "10s""#),
            "with_windows: half_life is a duration, which needs a temporal clock column \
                 (Datetime, Date or Duration), but clock is \"f\" (Float64)",
        ),
        (
            call(r#""1s""#, ""),
            "with_windows: half_life is a duration, which needs a temporal clock column \
                 (Datetime, Date or Duration), but clock is not given",
        ),
        (
            call("1", r#", "clock": "t", "gap_cap": 10"#),
            "with_windows: half_life is a number, but clock column \"t\" is Datetime(\"ms\"), \
                 which measures time. Give half_life as a duration",
        ),
        (
            call(r#""3d""#, r#", "clock": "d", "gap_cap": "1h""#),
            "with_windows: gap_cap is 1h, less than a day, the smallest step clock column \
                 \"d\" (Date) can take, so every step would be cut to it",
        ),
        (
            call(
                r#""3d""#,
                r#", "clock": "d", "gap_cap": "5d", "restart_after_step_back": "1h""#,
            ),
            "with_windows: restart_after_step_back is 1h, less than a day, the smallest step \
                 clock column \"d\" (Date) can take, so no step back could be as small and every \
                 one would reset the windows. Set restart_after_step_back to at least a day, or \
                 to 0 to reset at every step back.",
        ),
    ] {
        let err = WindowsRun::new(config(&json), df.schema())
            .err()
            .unwrap_or_else(|| panic!("{json}: ran"));
        assert!(err.contains(says), "{json}: {err}");
    }
    // A day and more is a step a `Date` clock can take.
    let ok = call(
        r#""3d""#,
        r#", "clock": "d", "gap_cap": "5d", "restart_after_step_back": "1d""#,
    );
    let out = run_whole(&ok, &df);
    assert_eq!(values_of(&out, "m").len(), 2);
}

/// Task 270 (review round 9, F-1): two finite clock values whose
/// difference is past the largest double are refused by row, naming
/// both, where the step was taken at the cap and left every later
/// stamp's removed time infinite.
#[test]
fn a_step_past_the_largest_double_is_refused_by_row() {
    let cfg = r#"{"formulas": [{"name": "m", "tree": ["ewm_sum", ["col", "x"],
            {"half_life": 10, "window_size": 3}]}], "clock": "t", "gap_cap": 100}"#;
    let df = df!("t" => [-1e308, 1e308, 1e308], "x" => [1.0, 2.0, 3.0]).unwrap();
    let mut run = WindowsRun::new(config(cfg), df.schema()).unwrap();
    let err = run.feed(&df, None).unwrap_err().to_string();
    assert!(
        err.contains("with_windows: clock column \"t\" steps from")
            && err.contains("at row 1 (counting from 0), a difference past the largest float"),
        "{err}"
    );
}

/// Task 252: an integer clock is read as its integers: a null on it is
/// refused by row, an unsigned value past an `Int64` by row with the
/// ceiling, and a step back by its size in integers; a refused first
/// row of a chunk leaves the run where the chunk before left it.
#[test]
fn an_integer_clock_is_refused_by_row_in_its_own_integers() {
    let cfg = r#"{"formulas": [{"name": "m", "tree": ["ewm_sum", ["col", "x"],
            {"half_life": 10, "window_size": 3}]}], "clock": "t", "gap_cap": 100}"#;
    let refused = |df: DataFrame| {
        let mut run = WindowsRun::new(config(cfg), df.schema()).unwrap();
        run.feed(&df, None).unwrap_err().to_string()
    };
    let nulls = df!("t" => [Some(1i64), None, Some(3)], "x" => [1.0, 2.0, 3.0]).unwrap();
    let err = refused(nulls);
    assert!(
        err.contains("with_windows: clock column \"t\" is null, NaN or infinite at row 1"),
        "{err}"
    );
    let past = df!("t" => [1u64, u64::MAX], "x" => [1.0, 2.0]).unwrap();
    let err = refused(past);
    assert!(
        err.contains(&format!(
            "with_windows: clock column \"t\" is {} at row 1 (counting from 0), past \
                 9223372036854775807, the largest value an integer clock holds (an Int64's)",
            u64::MAX
        )),
        "{err}"
    );
    // Near 2^62, where a double resolves 1024: the step back is 3, said
    // as the integers' difference.
    const T0: i64 = 1 << 62;
    let df = df!(
        "t" => [T0, T0 + 5, T0 + 9, T0 + 6, T0 + 10],
        "x" => [1.0, 2.0, 4.0, 8.0, 16.0],
    )
    .unwrap();
    let mut run = WindowsRun::new(config(cfg), df.schema()).unwrap();
    let mut out = run.feed(&df.slice(0, 3), None).unwrap();
    let err = run.feed(&df.slice(3, 2), None).unwrap_err().to_string();
    assert!(
        err.contains("clock column \"t\" steps back by 3 at row 3"),
        "{err}"
    );
    // The refused chunk changed nothing: the chunk that follows the
    // first in clock order goes on as one run of both would.
    let next = df!("t" => [T0 + 10], "x" => [16.0]).unwrap();
    out.vstack_mut(&run.feed(&next, None).unwrap()).unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    let mut whole = df.slice(0, 3);
    whole.vstack_mut(&next).unwrap();
    assert!(out.equals_missing(&run_whole(cfg, &whole)));
    // Sums over `(t - 3, t]`, from the definition: weights 2^(-d/10).
    let got = values_of(&out, "m");
    assert_eq!(got[..3], [Some(1.0), Some(2.0), Some(4.0)]);
    let last = got[3].unwrap();
    let want = 16.0 + 4.0 * 0.5f64.powf(0.1);
    assert!((last - want).abs() < 1e-12, "{last} vs {want}");
}

/// Task 252: a step back larger than `restart_after_step_back` starts
/// an increment over -- the step taken in integers on an integer clock
/// and in integer nanoseconds on a temporal one -- so the row after it
/// has no increment, and the next one steps from it.
#[test]
fn a_restart_starts_the_increments_over_on_integer_and_temporal_clocks() {
    let x = [1.0, 2.0, 4.0, 8.0, 16.0];
    let want = [None, Some(1.0), Some(2.0), None, Some(8.0)];
    // An integer clock near 2^62: steps back of 7 restart under 2.
    const T0: i64 = 1 << 62;
    let int = df!("t" => [T0 + 10, T0 + 11, T0 + 12, T0 + 5, T0 + 6], "x" => x).unwrap();
    let cfg = format!(
        r#"{{"formulas": [{}], "clock": "t", "gap_cap": 100,
            "restart_after_step_back": 2}}"#,
        increment_formula(r#"["col", "x"]"#, "1000", "0.5")
    );
    assert_eq!(values_of(&run_whole(&cfg, &int), "d"), want);
    // A temporal clock: a step back of 7 s restarts under 2 s.
    let ns = Series::new("t".into(), [10i64, 11, 12, 5, 6].map(|s| s * 1_000_000_000))
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap();
    let temporal = df!("t" => ns, "x" => x).unwrap();
    let cfg = format!(
        r#"{{"formulas": [{}], "clock": "t", "gap_cap": "100s",
            "restart_after_step_back": "2s"}}"#,
        increment_formula(r#"["col", "x"]"#, r#""1000s""#, r#""500ms""#)
    );
    assert_eq!(values_of(&run_whole(&cfg, &temporal), "d"), want);
}

/// Task 252: a null input of an integer or a temporal increment is no
/// step: its row's increment is null, and the next row steps from the
/// last value -- an integer one in integers, a temporal one in seconds.
#[test]
fn a_null_integer_or_temporal_input_is_no_step() {
    let at = Series::new("at".into(), [Some(0i64), None, Some(2_500), Some(4_000)])
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        .unwrap();
    let df = df!(
        "t" => [0.0, 1.0, 2.0, 3.0],
        "y" => [Some(1i64), None, Some(4), Some(10)],
        "at" => at,
    )
    .unwrap();
    let cfg = format!(
        r#"{{"formulas": [{}, {}], "clock": "t", "gap_cap": 100}}"#,
        increment_formula(r#"["col", "y"]"#, "1000", "0.5"),
        increment_formula(r#"["col", "at"]"#, "1000", "0.5").replace("\"d\"", "\"e\"")
    );
    let out = run_whole(&cfg, &df);
    assert_eq!(values_of(&out, "d"), [None, None, Some(3.0), Some(6.0)]);
    assert_eq!(values_of(&out, "e"), [None, None, Some(2.5), Some(1.5)]);
}

/// Task 252: a null group key is a group of its own -- for the
/// increments as for the windows -- and keeps its increment across a
/// save and a load; the next file's first row, of that group, is put
/// to the clock policy as that group's (a step forward, which runs).
#[test]
fn a_null_group_keeps_its_increments_across_a_save() {
    let df = df!(
        "t" => [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "g" => [Some("a"), None, None, Some("a"), None, Some("a")],
        "x" => [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    )
    .unwrap();
    let cfg = format!(
        r#"{{"formulas": [{}], "clock": "t", "gap_cap": 100, "group": "g"}}"#,
        increment_formula(r#"["col", "x"]"#, "1000", "0.5")
    );
    // From the definition: each group's own previous value.
    let want = [None, None, Some(2.0), Some(7.0), Some(12.0), Some(24.0)];
    assert_eq!(values_of(&run_whole(&cfg, &df), "d"), want);
    let mut first = WindowsRun::new(config(&cfg), df.schema()).unwrap();
    let mut out = first.feed(&df.slice(0, 3), None).unwrap();
    let state = first.save_bytes().unwrap();
    let mut second = WindowsRun::load_bytes(&state, config(&cfg), df.schema()).unwrap();
    // The next file begins with the null group's row.
    let tail = df!(
        "t" => [4.0, 5.0],
        "g" => [None, Some("a")],
        "x" => [16.0, 32.0],
    )
    .unwrap();
    out.vstack_mut(&second.feed(&tail, None).unwrap()).unwrap();
    out.vstack_mut(&second.finish().unwrap()).unwrap();
    // Rows 0..3 and then 4, 5: the null group's 16 steps from its 4.
    assert_eq!(
        values_of(&out, "d"),
        [None, None, Some(2.0), Some(12.0), Some(31.0)]
    );
}

/// Task 252: what a run reports of its windows' cost, the rows their
/// queues hold: a window of 2 over rows a unit apart holds the two rows
/// of the window the last stamp resolved and the row that waits for the
/// next stamp, three; at the end of input, the last row's window, two,
/// kept for a next file.
#[test]
fn a_run_counts_the_rows_its_windows_hold() {
    let cfg = r#"{"formulas": [{"name": "m", "tree": ["ewm_sum", ["col", "x"],
            {"half_life": 10, "window_size": 2}]}], "clock": "t", "gap_cap": 100}"#;
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut run = WindowsRun::new(config(cfg), df.schema()).unwrap();
    assert_eq!(run.queued(), 0);
    run.feed(&df, None).unwrap();
    assert_eq!(run.queued(), 3);
    run.finish().unwrap();
    assert_eq!(run.queued(), 2);
}

/// Task 252: `save` writes what `save_bytes` returns, to a file a run
/// loads and goes on from as one run would.
#[test]
fn a_state_saved_to_a_file_resumes() {
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut first = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    let mut out = first.feed(&df.slice(0, 4), None).unwrap();
    let dir = std::env::temp_dir().join(format!("rheon-windows-save-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("state.msgpack");
    first.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(bytes, first.save_bytes().unwrap());
    let mut second = WindowsRun::load_bytes(&bytes, config(FORWARD), df.schema()).unwrap();
    out.vstack_mut(&second.feed(&df.slice(4, 2), None).unwrap())
        .unwrap();
    out.vstack_mut(&second.finish().unwrap()).unwrap();
    assert!(out.equals_missing(&run_whole(FORWARD, &df)));
}

/// Task 252: a state is refused, by name, when its body does not decode
/// though its header does, when its core is not its own call's, when it
/// holds another count of rows than its core, when the rows it holds
/// are of another schema than the frame, and when it was fed another
/// kind of clock than the frame's.
#[test]
fn a_state_that_does_not_fit_its_call_or_its_frame_is_refused() {
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut first = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    first.feed(&df.slice(0, 4), None).unwrap();
    let bytes = first.save_bytes().unwrap();
    let file: FileIn = rmp_serde::from_slice(&bytes).unwrap();
    let load = |b: &[u8]| {
        WindowsRun::load_bytes(b, config(FORWARD), df.schema())
            .err()
            .expect("refused")
    };
    // The header alone.
    #[derive(Serialize)]
    struct Only<'a> {
        magic: &'a str,
        version: u32,
    }
    let header = rmp_serde::to_vec_named(&Only {
        magic: LEGACY_WINDOWS_MAGIC,
        version: LAST_LEGACY_VERSION,
    })
    .unwrap();
    // Unsealed, it is refused as damaged before it is decoded; sealed,
    // by the decoder.
    let err = load(&header);
    assert!(
        err.starts_with("with_windows: the state is damaged (the checksum it ends with"),
        "{err}"
    );
    let err = load(&crate::reseal(&header));
    assert!(
        err.starts_with("with_windows: the state cannot be read ("),
        "{err}"
    );
    let encode = |core: &Windows, held: &[u8]| {
        rmp_serde::to_vec_named(&FileOut {
            magic: WINDOWS_MAGIC,
            kind: WINDOWS_KIND,
            version: WINDOWS_VERSION,
            config: &file.config,
            core,
            held,
            increments: &file.increments,
            stream_clock: file.stream_clock,
            key_dtypes: &file.key_dtypes,
            resume_skip: file.resume_skip,
            resume_first: file.resume_first,
            resume_first_session: file.resume_first_session,
            last_row: &file.last_row,
            checksum: None,
        })
        .map(|b| crate::reseal(&b))
        .unwrap()
    };
    // The core of another call: a window of 3 under this call's 2.
    let other = FORWARD.replace("\"window_size\": 2", "\"window_size\": 3");
    assert_ne!(other, FORWARD);
    let mut run3 = WindowsRun::new(config(&other), df.schema()).unwrap();
    run3.feed(&df.slice(0, 4), None).unwrap();
    let file3: FileIn = rmp_serde::from_slice(&run3.save_bytes().unwrap()).unwrap();
    let err = load(&encode(&file3.core, &file.held));
    assert_eq!(
        err,
        "with_windows: the state's window operators are not those of the call that saved \
             it, so the state is damaged. Load an undamaged copy of the state file, or leave out \
             load_state to start over."
    );
    // Held rows of another count than the core's.
    let mut fewer = IpcReader::new(Cursor::new(file.held.clone()))
        .finish()
        .unwrap();
    assert!(fewer.height() > 1);
    fewer = fewer.slice(0, 1);
    let mut ipc = Vec::new();
    IpcWriter::new(&mut ipc).finish(&mut fewer).unwrap();
    let err = load(&encode(&file.core, &ipc));
    assert_eq!(
        err,
        format!(
            "with_windows: the state holds 1 rows but counts {}, so it is damaged. Load an \
                 undamaged copy of the state file, or leave out load_state to start over.",
            file.core.held()
        )
    );
    // Held rows of another schema than the frame: `t` an integer.
    let int = df!(
        "t" => [0i64, 1, 2, 3, 4, 5],
        "x" => [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        "wide" => df.column("wide").unwrap().as_materialized_series().clone(),
    )
    .unwrap();
    let err = WindowsRun::load_bytes(&bytes, config(FORWARD), int.schema())
        .err()
        .expect("refused");
    assert!(
            err.contains(&format!(
                "with_windows: the state holds {} rows with columns [\"t: Float64\", \"x: Float64\", \
                 \"wide: String\"], but the input has [\"t: Int64\", \"x: Float64\", \"wide: String\"]",
                file.core.held()
            )),
            "{err}"
        );
    // A state that holds no row loads on that frame, and its first chunk
    // is refused there: it was fed a float clock.
    let mut done = WindowsRun::new(config(FORWARD), df.schema()).unwrap();
    done.feed(&df.slice(0, 4), None).unwrap();
    done.finish().unwrap();
    assert_eq!(done.held(), 0);
    let mut resumed =
        WindowsRun::load_bytes(&done.save_bytes().unwrap(), config(FORWARD), int.schema()).unwrap();
    let err = resumed
        .feed(&int.slice(4, 2), None)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(
            "with_windows: clock column \"t\" is Int64, but the state it resumes read a \
                 float one. Resume the state on a clock of the kind it was saved with."
        ),
        "{err}"
    );
    // And the other way: a state fed an integer clock refuses a float one.
    let mut done = WindowsRun::new(config(FORWARD), int.schema()).unwrap();
    done.feed(&int.slice(0, 4), None).unwrap();
    done.finish().unwrap();
    let mut resumed =
        WindowsRun::load_bytes(&done.save_bytes().unwrap(), config(FORWARD), df.schema()).unwrap();
    let err = resumed.feed(&df.slice(4, 2), None).unwrap_err().to_string();
    assert!(
        err.contains(
            "with_windows: clock column \"t\" is Float64, but the state it resumes read an \
                 integer one"
        ),
        "{err}"
    );
}
