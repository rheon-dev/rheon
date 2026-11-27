//! A window run's state file read back, what its increments' part may hold
//! (the 2026-11-23 review, WIN-11), and a clock past the largest double
//! through the run's block paths (WIN-9; PLAN 578d).
use super::*;
use crate::windows::Stat;

const INCR: &str = r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["increment",
    ["col", "x"]], {"half_life": 4}]}], "clock": "t", "gap_cap": 100}"#;

fn frame(t: &[f64]) -> DataFrame {
    df!("t" => t, "x" => t.iter().map(|v| v * v).collect::<Vec<_>>()).unwrap()
}

/// The file `file` holds with `increments` in place of its own, sealed.
fn with_increments(file: &FileIn, increments: &[IncrState]) -> Vec<u8> {
    rmp_serde::to_vec_named(&FileOut {
        magic: WINDOWS_MAGIC,
        kind: WINDOWS_KIND,
        version: WINDOWS_VERSION,
        config: &file.config,
        core: &file.core,
        held: &file.held,
        increments,
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
}

/// WIN-11: a group's increment state holds one previous value per
/// increment in each of its three forms, and one state per key. A state
/// whose lists were short loaded, and the next row indexed past them and
/// panicked; one with two states for a key, or two for the null key,
/// loaded and read one of them. Each is refused as damaged at load.
#[test]
fn a_state_whose_increments_do_not_fit_its_call_is_refused() {
    let config = || serde_json::from_str::<WindowsConfig>(INCR).unwrap();
    let df = frame(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut first = WindowsRun::new(config(), df.schema()).unwrap();
    first.feed(&df.slice(0, 4), None).unwrap();
    let bytes = first.save_bytes().unwrap();
    let file: FileIn = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(file.increments.len(), 1, "the null key's state");
    let st = &file.increments[0];
    assert_eq!(st.prev.len(), 1);
    type Damage = (&'static str, fn(&mut Vec<IncrState>));
    let damage: [Damage; 6] = [
        ("prev short", |v| {
            v[0].prev.clear();
        }),
        ("prev_ns short", |v| {
            v[0].prev_ns.clear();
        }),
        ("prev_int short", |v| {
            v[0].prev_int.clear();
        }),
        ("prev long", |v| v[0].prev.push(1.0)),
        ("two null keys", |v| {
            let again = v[0].clone();
            v.push(again);
        }),
        ("two of one key", |v| {
            v[0].key = Some("a".into());
            let again = v[0].clone();
            v.push(again);
        }),
    ];
    for (what, f) in damage {
        let mut incr = file.increments.clone();
        f(&mut incr);
        let damaged = with_increments(&file, &incr);
        match WindowsRun::load_bytes(&damaged, config(), df.schema()) {
            Err(err) => assert!(err.contains("the state is damaged"), "{what}: {err}"),
            Ok(_) => panic!("{what}: loaded"),
        }
    }
    // The file as saved, through the same encoding, loads and goes on.
    let whole = with_increments(&file, &file.increments);
    let mut run = WindowsRun::load_bytes(&whole, config(), df.schema()).unwrap();
    run.feed(&df.slice(4, 2), None).unwrap();
}

/// WIN-9 through a frame: the blocks a run takes its rows in, one group and
/// several, end at a row whose distance from its windows' first row passes
/// the largest double, and the row is refused by name, as one pushed alone.
#[test]
fn a_clock_past_the_largest_double_is_refused_through_each_block_path() {
    for grouped in [false, true] {
        let json = format!(
            r#"{{"formulas": [{{"name": "f", "tree": ["ewm_mean", ["col", "x"],
            {{"half_life": 10, "window_size": 50}}]}}], "clock": "t", "gap_cap": 1e308{}}}"#,
            if grouped { r#", "group": "g""# } else { "" }
        );
        let config = serde_json::from_str::<WindowsConfig>(&json).unwrap();
        let df = df!(
            "t" => [-1e308, 0.0, 1e308],
            "x" => [10.0, 20.0, 30.0],
            "g" => ["a", "a", "a"],
        )
        .unwrap();
        let mut run = WindowsRun::new(config, df.schema()).unwrap();
        let err = run.feed(&df, None).expect_err("refused").to_string();
        assert!(
            err.contains("has run from") && err.contains("largest float"),
            "grouped {grouped}: {err}"
        );
    }
}

/// WIN-4/5/10 (option (b)) through `WindowsRun`: exercises the plain fast path
/// (unbounded mean, variance, rate), the windowed backward queue path and
/// forward close at extreme half-lives (`1e200`, `1.5e308`, `f64::MAX`), and
/// verifies bit-exact chunk invariance and state save/load continuation with
/// no `WINDOWS_VERSION` bump.
#[test]
fn extreme_half_lives_save_load_and_continue_to_the_bit() {
    let df = df!(
        "t" => [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0, 13.0],
        "x" => [Some(10.0), Some(20.0), Some(30.0), None, Some(40.0), Some(50.0), Some(60.0), Some(70.0)],
    )
    .unwrap();
    for h in [1e200, 1.5e308, f64::MAX] {
        // Exact decimal serialization of `h` so serde_json recovers the exact `f64`.
        let json = format!(
            r#"{{
                "formulas": [
                    {{"name": "um", "tree": ["ewm_mean", ["col", "x"], {{"half_life": {h}}}]}},
                    {{"name": "uv", "tree": ["ewm_var",  ["col", "x"], {{"half_life": {h}}}]}},
                    {{"name": "ur", "tree": ["ewm_rate", ["col", "x"], {{"half_life": {h}}}]}},
                    {{"name": "wm", "tree": ["ewm_mean", ["col", "x"], {{"half_life": {h}, "window_size": 10.0}}]}},
                    {{"name": "ws", "tree": ["ewm_std",  ["col", "x"], {{"half_life": {h}, "window_size": 10.0}}]}},
                    {{"name": "wr", "tree": ["ewm_rate", ["col", "x"], {{"half_life": {h}, "window_size": 10.0}}]}},
                    {{"name": "fm", "tree": ["rewm_mean", ["col", "x"], {{"half_life": {h}, "window_size": 10.0}}]}},
                    {{"name": "fr", "tree": ["rewm_rate", ["col", "x"], {{"half_life": {h}, "window_size": 10.0}}]}}
                ],
                "clock": "t",
                "gap_cap": 1e9
            }}"#
        );
        let config = || serde_json::from_str::<WindowsConfig>(&json).unwrap();
        // Whole frame in one shot (exercises `plain.rs` fast path for unbounded ops).
        let mut whole = WindowsRun::new(config(), df.schema()).unwrap();
        let mut out_whole = whole.feed(&df, None).unwrap();
        out_whole.vstack_mut(&whole.finish().unwrap()).unwrap();

        // Check expected values at row 2 (t = 2.0): mean 10, var 125, rate 30,
        // windowed mean 13, windowed std sqrt(410*10/34), windowed rate 30.
        let get = |frame: &DataFrame, col: &str, row: usize| -> f64 {
            frame
                .column(col)
                .unwrap()
                .f64()
                .unwrap()
                .get(row)
                .unwrap_or(f64::NAN)
        };
        let near = |got: f64, want: f64, what: &str| {
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "h {h} {what}: {got} against {want}"
            );
        };
        near(get(&out_whole, "um", 2), 10.0, "unbounded mean");
        near(get(&out_whole, "uv", 2), 125.0, "unbounded var");
        near(get(&out_whole, "ur", 2), 30.0, "unbounded rate");
        near(get(&out_whole, "wm", 2), 13.0, "windowed mean");
        near(
            get(&out_whole, "ws", 2),
            (410.0 * 10.0 / 34.0f64).sqrt(),
            "windowed std",
        );
        near(get(&out_whole, "wr", 2), 30.0, "windowed rate");
        assert!(get(&out_whole, "fm", 0).is_finite(), "h {h}: fwd mean");
        assert!(get(&out_whole, "fr", 0).is_finite(), "h {h}: fwd rate");

        // Split across 3 chunks with save/load after chunk 1: bit-identical
        // outputs and re-saved bytes equal loaded bytes.
        let mut part = WindowsRun::new(config(), df.schema()).unwrap();
        let mut out_split = part.feed(&df.slice(0, 3), None).unwrap();
        let saved = part.save_bytes().unwrap();
        let mut resumed = WindowsRun::load_bytes(&saved, config(), df.schema()).unwrap();
        assert_eq!(resumed.save_bytes().unwrap(), saved, "h {h}: re-save bytes");
        out_split
            .vstack_mut(&resumed.feed(&df.slice(3, 2), None).unwrap())
            .unwrap();
        out_split
            .vstack_mut(&resumed.feed(&df.slice(5, 3), None).unwrap())
            .unwrap();
        out_split.vstack_mut(&resumed.finish().unwrap()).unwrap();

        assert_eq!(out_whole.height(), out_split.height());
        for col in ["um", "uv", "ur", "wm", "ws", "wr", "fm", "fr"] {
            let a = out_whole.column(col).unwrap().f64().unwrap();
            let b = out_split.column(col).unwrap().f64().unwrap();
            for r in 0..out_whole.height() {
                assert_eq!(
                    a.get(r).map(f64::to_bits),
                    b.get(r).map(f64::to_bits),
                    "h {h} col {col} row {r}: {:?} vs {:?}",
                    a.get(r),
                    b.get(r)
                );
            }
        }
    }
}

/// Build the legacy v9..=14 placeholder core (`1 kernel, 1 Sum op`, task 78)
/// and step it through `df` identically to a v14 row-only `WindowsRun`, so a
/// test can encode genuine v9..=14 states carrying the old stand-in kernel and
/// queued placeholder rows.
fn build_legacy_placeholder_core(cfg: &WindowsConfig, df: &DataFrame) -> Windows {
    let clock_cfg = crate::spec::clock_cfg_of(&crate::spec::ClockPolicy {
        who: WHO,
        clock: cfg.clock.as_deref(),
        gap_cap: cfg.gap_cap.as_ref(),
        restart_after_step_back: cfg.restart_after_step_back.as_ref(),
        session: cfg.session.as_deref(),
        session_gap: cfg.session_gap.as_ref(),
        spec_closes_on_session: None,
        group: cfg.group.as_deref(),
        looks_ahead: false,
    })
    .unwrap();
    let mut core = Windows::new(
        vec![KernelDef {
            direction: Direction::Backward,
            half_life: f64::INFINITY,
            window_size: Some(1.0),
            window_ns: Some(1_000_000_000),
            closed: Closed::Left,
            since: None,
        }],
        vec![OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
            quantile: None,
            max_state_bytes: None,
            interpolation: None,
            rank_method: None,
        }],
        clock_cfg,
    )
    .unwrap();
    core.set_grouped(cfg.group.is_some());
    let n = df.height();
    let clock: Option<Vec<ClockValue>> = cfg.clock.as_deref().map(|c| {
        let s = df.column(c).unwrap().as_materialized_series();
        match s.dtype() {
            DataType::Int64 => s
                .i64()
                .unwrap()
                .iter()
                .map(|v| ClockValue::I64(v.unwrap()))
                .collect(),
            DataType::Float64 => s
                .f64()
                .unwrap()
                .iter()
                .map(|v| ClockValue::F64(v.unwrap()))
                .collect(),
            DataType::Datetime(..) => s
                .to_physical_repr()
                .i64()
                .unwrap()
                .iter()
                .map(|v| ClockValue::Ns(v.unwrap()))
                .collect(),
            other => panic!("unexpected test clock dtype {other:?}"),
        }
    });
    let session: Option<Vec<u64>> = cfg
        .session
        .as_deref()
        .map(|c| session_hashes(df.column(c).unwrap().as_materialized_series()).unwrap());
    let group_col = cfg
        .group
        .as_deref()
        .map(|c| key_text(df.column(c).unwrap().as_materialized_series()).unwrap());
    let groups = group_col.as_ref().map(|c| c.str().unwrap());
    for i in 0..n {
        let key = match groups {
            None => Some(""),
            Some(g) => g.get(i),
        };
        let gi = core.group(key);
        core.push(&RowIn {
            group: gi,
            clock: clock.as_ref().map(|c| c[i]),
            session: session.as_ref().map(|s| s[i]),
            values: &[0.0],
            accept: true,
            lists: &[],
        })
        .unwrap();
    }
    let _ = core.drain();
    assert!(
        core.queued() > 0,
        "legacy placeholder queues at least one row"
    );
    core
}

/// Encode `file` with `core` and `version`: legacy magic for `v <= 11`
/// (unsealed for `v < 11`, sealed for `v == 11`) and enveloped sealed for
/// `v >= 12`.
fn encode_version(file: &FileIn, core: &Windows, version: u32) -> Vec<u8> {
    let enveloped = rmp_serde::to_vec_named(&FileOut {
        magic: WINDOWS_MAGIC,
        kind: WINDOWS_KIND,
        version,
        config: &file.config,
        core,
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
    if version <= LAST_LEGACY_VERSION {
        crate::envelope::legacy::windows(&enveloped, version)
    } else {
        enveloped
    }
}

/// Row-only calls (windows state 15, task 78): carry zero kernels and zero
/// window operators (no placeholder sum kernel, zero queued rows), match
/// plain-run and row-by-row push to the bit in outputs and saved state bytes,
/// preserve chunk invariance, sliced resume, gap_cap, session, step_back and
/// WIN-9 span refusals.
#[test]
fn row_only_calls_carry_zero_kernels_and_preserve_every_invariant() {
    let df = df!(
        "t" => [0.0, 0.4, 0.8, 0.8, 1.2, 20.0, 20.4, 21.0, 15.0, 15.5, 16.0, 16.5],
        "ev" => [-0.1, 0.2, 0.5, 0.5, 0.9, 19.8, 20.1, 20.6, 14.7, 15.1, 15.5, 16.0],
        "x" => [1.0, 1.0, 2.0, 2.0, 2.0, 5.0, 5.0, 3.0, 3.0, 4.0, 4.0, 1.0],
        "g" => ["a", "b", "a", "a", "b", "a", "b", "a", "b", "a", "b", "a"],
        "s" => ["s1", "s1", "s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2", "s2", "s2"],
    )
    .unwrap();

    let row_only_cfgs = [
        // Ungrouped with clock, gap_cap, session, restart_after_step_back
        r#"{"formulas": [
            {"name": "cs", "tree": ["clock_step", ["col", "t"], {"half_life": 5.0, "stat": "mean"}]},
            {"name": "inc", "tree": ["increment", ["col", "x"]]},
            {"name": "rl", "tree": ["run_length", ["col", "x"]]},
            {"name": "dr", "tree": ["clock_drift", ["col", "t"], {"event": ["col", "ev"], "half_life": 5.0}]},
            {"name": "ts", "tree": ["tick_sign", ["col", "x"]]}
        ], "clock": "t", "gap_cap": 10.0, "session": "s", "restart_after_step_back": 2.0}"#,
        // Grouped with clock, gap_cap, session, restart_after_step_back
        r#"{"formulas": [
            {"name": "cs", "tree": ["clock_step", ["col", "t"], {"half_life": 5.0, "stat": "mean"}]},
            {"name": "inc", "tree": ["increment", ["col", "x"]]},
            {"name": "rl", "tree": ["run_length", ["col", "x"]]},
            {"name": "dr", "tree": ["clock_drift", ["col", "t"], {"event": ["col", "ev"], "half_life": 5.0}]},
            {"name": "ts", "tree": ["tick_sign", ["col", "x"]]}
        ], "clock": "t", "gap_cap": 10.0, "group": "g", "session": "s", "restart_after_step_back": 2.0}"#,
        // Row-count clock (no clock column), grouped and ungrouped
        r#"{"formulas": [
            {"name": "inc", "tree": ["increment", ["col", "x"]]},
            {"name": "rl", "tree": ["run_length", ["col", "x"]]},
            {"name": "ts", "tree": ["tick_sign", ["col", "x"]]}
        ], "session": "s"}"#,
        r#"{"formulas": [
            {"name": "inc", "tree": ["increment", ["col", "x"]]},
            {"name": "rl", "tree": ["run_length", ["col", "x"]]},
            {"name": "ts", "tree": ["tick_sign", ["col", "x"]]}
        ], "group": "g", "session": "s"}"#,
    ];

    for (ci, json) in row_only_cfgs.iter().enumerate() {
        let cfg = || serde_json::from_str::<WindowsConfig>(json).unwrap();
        // Plain loop ON vs OFF: outputs and saved state bytes must match byte-for-byte.
        let mut plain_on = WindowsRun::new(cfg(), df.schema()).unwrap();
        let mut plain_off = WindowsRun::new(cfg(), df.schema()).unwrap();
        plain_off.set_plain_run(false);

        let out_on = plain_on.feed(&df, None).unwrap();
        let out_off = plain_off.feed(&df, None).unwrap();
        assert!(out_on.equals_missing(&out_off), "case {ci}: plain vs push");
        assert_eq!(plain_on.queued(), 0, "case {ci}: zero queued rows");
        assert_eq!(plain_on.held(), 0, "case {ci}: zero held rows");
        assert!(plain_on.plain_rows() > 0, "case {ci}: plain loop activated");

        let bytes_on = plain_on.save_bytes().unwrap();
        let bytes_off = plain_off.save_bytes().unwrap();
        assert_eq!(bytes_on, bytes_off, "case {ci}: plain vs push state bytes");

        // Decoded state carries 0 kernels, 0 ops, 0 outputs, empty group queues.
        let file: FileIn = rmp_serde::from_slice(&bytes_on).unwrap();
        assert!(file.core.is_clock_only(), "case {ci}: clock-only core");
        assert!(file.core.kernels().is_empty(), "case {ci}: 0 kernels");
        assert!(file.core.ops().is_empty(), "case {ci}: 0 ops");
        assert_eq!(file.core.n_outputs(), 0, "case {ci}: 0 outputs");
        assert_eq!(file.core.queued(), 0, "case {ci}: 0 queued");

        // Chunk invariance + mid-stream save/load.
        let mut split = WindowsRun::new(cfg(), df.schema()).unwrap();
        let mut out_split = split.feed(&df.slice(0, 5), None).unwrap();
        let mid = split.save_bytes().unwrap();
        let mut resumed = WindowsRun::load_bytes(&mid, cfg(), df.schema()).unwrap();
        assert_eq!(resumed.save_bytes().unwrap(), mid, "case {ci}: re-save");
        for r in 5..df.height() {
            out_split
                .vstack_mut(&resumed.feed(&df.slice(r as i64, 1), None).unwrap())
                .unwrap();
        }
        assert!(out_on.equals_missing(&out_split), "case {ci}: chunk/resume");

        // Sliced resume: holds no rows, identifies input by last_row, refuses another.
        let mut sliced = WindowsRun::new(cfg(), df.schema()).unwrap();
        let mut out_sliced = sliced.feed(&df, Some(4)).unwrap();
        assert_eq!(
            (out_sliced.height(), sliced.consumed(), sliced.held()),
            (4, 4, 0)
        );
        let sl_bytes = sliced.save_bytes_with(sliced.consumed(), false).unwrap();
        let mut sl_resumed = WindowsRun::load_bytes(&sl_bytes, cfg(), df.schema()).unwrap();
        out_sliced
            .vstack_mut(&sl_resumed.feed(&df, None).unwrap())
            .unwrap();
        assert!(
            out_on.equals_missing(&out_sliced),
            "case {ci}: sliced resume"
        );
    }

    // Refused step-back and WIN-9 span overflow past f64::MAX still fire on row-only calls.
    for grouped in [false, true] {
        let step_json = format!(
            r#"{{"formulas": [{{"name": "cs", "tree": ["clock_step", ["col", "t"],
            {{"half_life": 5.0, "stat": "mean"}}]}}], "clock": "t", "gap_cap": 1e308{}}}"#,
            if grouped { r#", "group": "g""# } else { "" }
        );
        let cfg = || serde_json::from_str::<WindowsConfig>(&step_json).unwrap();
        let back_df = df!("t" => [0.0, 2.0, 1.0], "g" => ["a", "a", "a"]).unwrap();
        let mut run = WindowsRun::new(cfg(), back_df.schema()).unwrap();
        let err = run
            .feed(&back_df, None)
            .expect_err("step back refused")
            .to_string();
        assert!(err.contains("steps back"), "grouped {grouped}: {err}");

        let span_df = df!("t" => [-1e308, 0.0, 1e308], "g" => ["a", "a", "a"]).unwrap();
        let mut run = WindowsRun::new(cfg(), span_df.schema()).unwrap();
        let err = run
            .feed(&span_df, None)
            .expect_err("WIN-9 refused")
            .to_string();
        assert!(
            err.contains("has run from") && err.contains("largest float"),
            "grouped {grouped}: {err}"
        );
    }
}

/// Loader for windows state versions 9 to 14 on row-only calls: a legacy
/// state carrying the one-kernel, one-sum placeholder loads cleanly, strips
/// the placeholder so `queued() == 0`, re-saves byte-for-byte as a clean v15
/// state, and continues bit-identically; damaged states (v15 with placeholder,
/// v14 without placeholder, v14 with wrong kernel or corrupt queue, non-row
/// call with 0 kernels) are all refused.
#[test]
fn legacy_v9_to_v14_row_only_states_load_continue_to_the_bit_and_resave_without_placeholder() {
    let df = df!(
        "t" => [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3],
        "ev" => [-0.1, 0.1, 0.4, 0.6, 0.9, 1.1, 1.4, 1.7, 2.0, 2.2, 2.5, 2.8],
        "x" => [10.0, 10.0, 12.0, 12.0, 15.0, 14.0, 14.0, 18.0, 17.0, 17.0, 20.0, 22.0],
        "g" => ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
    )
    .unwrap();
    let first = df.slice(0, 6);
    let second = df.slice(6, 6);

    let configs = [
        // Increment + tick_sign (existed in v9..v14)
        r#"{"formulas": [
            {"name": "inc", "tree": ["increment", ["col", "x"]]},
            {"name": "ts", "tree": ["tick_sign", ["col", "x"]]}
        ], "clock": "t", "gap_cap": 100.0}"#,
        // Grouped clock_step + run_length + clock_drift (v14)
        r#"{"formulas": [
            {"name": "cs", "tree": ["clock_step", ["col", "t"], {"half_life": 5.0, "stat": "mean"}]},
            {"name": "rl", "tree": ["run_length", ["col", "x"]]},
            {"name": "dr", "tree": ["clock_drift", ["col", "t"], {"event": ["col", "ev"], "half_life": 5.0}]}
        ], "clock": "t", "gap_cap": 100.0, "group": "g"}"#,
        // Row-count clock (no clock column)
        r#"{"formulas": [
            {"name": "inc", "tree": ["increment", ["col", "x"]]},
            {"name": "rl", "tree": ["run_length", ["col", "x"]]}
        ]}"#,
    ];

    for (ci, json) in configs.iter().enumerate() {
        let cfg = || serde_json::from_str::<WindowsConfig>(json).unwrap();
        let mut whole = WindowsRun::new(cfg(), df.schema()).unwrap();
        let want_all = whole.feed(&df, None).unwrap();
        let want_second = want_all.slice(6, 6);

        let mut v15 = WindowsRun::new(cfg(), first.schema()).unwrap();
        v15.feed(&first, None).unwrap();
        let v15_bytes = v15.save_bytes().unwrap();
        let v15_file: FileIn = rmp_serde::from_slice(&v15_bytes).unwrap();
        let legacy_core = build_legacy_placeholder_core(&cfg(), &first);
        assert!(legacy_core.is_legacy_placeholder());

        // Versions 9..=14 (legacy unsealed, legacy sealed, enveloped sealed)
        // all load, strip the placeholder, re-save as exact v15 bytes, and
        // continue bit-identically.
        for version in 9..=14 {
            let legacy_bytes = encode_version(&v15_file, &legacy_core, version);
            let mut loaded = WindowsRun::load_bytes(&legacy_bytes, cfg(), df.schema())
                .unwrap_or_else(|e| {
                    panic!("case {ci} v{version}: {e}");
                });
            assert_eq!(
                loaded.queued(),
                0,
                "case {ci} v{version}: placeholder stripped"
            );
            assert_eq!(
                loaded.save_bytes().unwrap(),
                v15_bytes,
                "case {ci} v{version}: exact conversion to v15 bytes"
            );
            let got = loaded.feed(&second, None).unwrap();
            assert_eq!(got.height(), want_second.height());
            for f in &cfg().formulas {
                let a = got.column(&f.name).unwrap().f64().unwrap();
                let b = want_second.column(&f.name).unwrap().f64().unwrap();
                for r in 0..got.height() {
                    assert_eq!(
                        a.get(r).map(f64::to_bits),
                        b.get(r).map(f64::to_bits),
                        "case {ci} v{version} col {} row {r}",
                        f.name
                    );
                }
            }
        }

        // Damaged states must be refused:
        // 1) A v15 row-only file carrying the old placeholder kernel.
        let v15_with_placeholder = encode_version(&v15_file, &legacy_core, WINDOWS_VERSION);
        let err = WindowsRun::load_bytes(&v15_with_placeholder, cfg(), df.schema())
            .err()
            .expect("v15 with placeholder refused");
        assert!(err.contains("damaged"), "case {ci} v15 placeholder: {err}");

        // 2) A v14 row-only file with an empty core (no v14 build wrote one).
        let v14_empty = encode_version(&v15_file, &v15_file.core, LAST_PLACEHOLDER_VERSION);
        let err = WindowsRun::load_bytes(&v14_empty, cfg(), df.schema())
            .err()
            .expect("v14 empty core refused");
        assert!(err.contains("damaged"), "case {ci} v14 empty: {err}");

        // 3) A v14 row-only file whose placeholder core is damaged (corrupt queue).
        let mut damaged_core = legacy_core.clone();
        damaged_core.damage("queue");
        let v14_bad_queue = encode_version(&v15_file, &damaged_core, LAST_PLACEHOLDER_VERSION);
        let err = WindowsRun::load_bytes(&v14_bad_queue, cfg(), df.schema())
            .err()
            .expect("v14 damaged placeholder queue refused");
        assert!(err.contains("damaged"), "case {ci} v14 bad queue: {err}");
    }

    // 4) A v14 row-only file carrying a non-placeholder kernel (window_size 2.0).
    let row_cfg = serde_json::from_str::<WindowsConfig>(configs[0]).unwrap();
    let mean_cfg = serde_json::from_str::<WindowsConfig>(
        r#"{"formulas": [{"name": "m", "tree": ["ewm_mean", ["col", "x"], {"half_life": 5.0}]}],
            "clock": "t", "gap_cap": 100.0}"#,
    )
    .unwrap();
    let mut row_run = WindowsRun::new(row_cfg.clone(), first.schema()).unwrap();
    row_run.feed(&first, None).unwrap();
    let row_file: FileIn = rmp_serde::from_slice(&row_run.save_bytes().unwrap()).unwrap();
    let mut mean_run = WindowsRun::new(mean_cfg.clone(), first.schema()).unwrap();
    mean_run.feed(&first, None).unwrap();
    let mean_file: FileIn = rmp_serde::from_slice(&mean_run.save_bytes().unwrap()).unwrap();

    let v14_wrong_kernel = encode_version(&row_file, &mean_file.core, LAST_PLACEHOLDER_VERSION);
    let err = WindowsRun::load_bytes(&v14_wrong_kernel, row_cfg, df.schema())
        .err()
        .expect("v14 wrong kernel refused");
    assert!(err.contains("damaged"), "v14 wrong kernel: {err}");

    // 5) A v15 non-row-only file tampered to have a 0-kernel clock-only core.
    let v15_missing_ops = encode_version(&mean_file, &row_file.core, WINDOWS_VERSION);
    let err = WindowsRun::load_bytes(&v15_missing_ops, mean_cfg, df.schema())
        .err()
        .expect("v15 empty core for window op refused");
    assert!(err.contains("damaged"), "v15 missing ops: {err}");
}
