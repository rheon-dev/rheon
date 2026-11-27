//! A window run's state file read back, what its increments' part may hold
//! (the 2026-11-23 review, WIN-11), and a clock past the largest double
//! through the run's block paths (WIN-9; PLAN 578d).
use super::*;

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
