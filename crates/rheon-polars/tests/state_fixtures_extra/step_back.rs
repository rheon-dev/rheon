//! Frozen state fixtures for per-group step-back skip (`on_step_back = "skip"`)
//! across the bank, `with_windows`, `merge`, `refresh_time`, and `bars`
//! (hard rule 5: every new persisted form held non-empty in its fixture).

use super::*;

fn bank_step_back_specs() -> Vec<Spec> {
    [
        r#"{"name": "skip_only", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0, "group": "g", "on_step_back": "skip"}"#,
        r#"{"name": "skip_or_reset", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0, "group": "g", "on_step_back": "skip", "restart_after_step_back": 5.0}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

fn bank_step_back_frame() -> DataFrame {
    // Two interleaved groups: "a" has stale rows (step back <= 5) and a large
    // step back (> 5) before the cut; "b" stays monotonic before the cut and
    // has stale rows after the cut.
    let g = [
        "a", "b", "a", "b", "a", "b", "a", "b", "a", "b", // 0..10 (before cut)
        "a", "b", "a", "b", "a", "b", "a", "b", "a", "b", // 10..20 (after cut)
    ];
    let t = [
        10.0, 10.0, 14.0, 12.0, 12.0, // row 4: a steps back by 2 (skip)
        15.0, 16.0, 18.0,
        8.0,  // row 8: a steps back by 8 (skip in skip_only, reset in skip_or_reset)
        20.0, // row 9: b at 20.0
        18.0, 17.0, // row 11: b steps back by 3 (skip)
        17.0, // row 12: a steps back by 1 (skip in both)
        22.0, 21.0, 25.0, 20.0, // row 16: a steps back by 1 (skip)
        27.0, 24.0, 30.0,
    ];
    let x0: Vec<f64> = (0..20).map(|i| (i as f64) * 0.3 - 1.5).collect();
    let x1: Vec<f64> = (0..20).map(|i| ((i % 5) as f64) - 2.0).collect();
    let y: Vec<f64> = x0
        .iter()
        .zip(&x1)
        .enumerate()
        .map(|(i, (a, b))| 0.5 + 1.2 * a - 0.7 * b + ((i % 3) as f64) * 0.05)
        .collect();
    df!("t" => t, "g" => g, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

pub(super) fn bank_step_back_skip_first() -> (Vec<u8>, DataFrame) {
    let df = bank_step_back_frame();
    let mut bank = Bank::new(bank_step_back_specs()).unwrap();
    bank.fit_predict(&df.slice(0, 10)).unwrap();
    let _ = bank.take_notices();
    (bank.save_bytes().unwrap(), df.slice(10, 10))
}

pub(super) fn bank_step_back_skip_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let specs = bank_step_back_specs();
    let mut bank = Bank::load_bytes(bytes, Some(&specs))?;
    let again = bank.save_bytes()?;
    let summary0_before = bank.summary(0, None).map_err(|e| e.to_string())?;
    let summary1_before = bank.summary(1, None).map_err(|e| e.to_string())?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    let summary0_after = bank.summary(0, None).map_err(|e| e.to_string())?;
    let summary1_after = bank.summary(1, None).map_err(|e| e.to_string())?;
    let notices = df!("notice" => bank.take_notices()).map_err(|e| e.to_string())?;
    Ok((
        again,
        vec![
            ("summary0_before".into(), summary0_before),
            ("summary1_before".into(), summary1_before),
            ("fit_predict".into(), out),
            ("summary0_after".into(), summary0_after),
            ("summary1_after".into(), summary1_after),
            ("notices".into(), notices),
        ],
    ))
}

fn windows_step_back_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "back", "tree": ["ewm_mean", ["col", "x"], {"half_life": 5.0, "window_size": 10.0}]},
               {"name": "ahead", "tree": ["rewm_mean", ["col", "x"], {"half_life": 5.0, "window_size": 10.0}]},
               {"name": "step", "tree": ["increment", ["col", "x"]]}
             ],
             "clock": "t", "gap_cap": 50.0, "group": "g", "on_step_back": "skip"}"#,
    )
    .unwrap()
}

fn windows_step_back_frame() -> DataFrame {
    // Group "a" has a stale row at t=8.0 inside the open forward window of
    // row 0 (t=10.0, window [10, 20]), so `stale_seqs` and
    // `Group::skipped_step_back` are both non-empty at the cut (row 6).
    df!(
        "t" => [10.0, 10.0, 12.0, 11.0, 14.0, 13.0, 18.0, 16.0, 22.0, 25.0],
        "g" => ["a",  "b",  "a",  "a",  "b",  "b",  "a",  "a",  "a",  "b"],
        "x" => [1.0,  10.0, 2.0,  99.0, 12.0, 88.0, 3.0,  77.0, 4.0,  14.0],
    )
    .unwrap()
}

pub(super) fn windows_step_back_skip_first() -> (Vec<u8>, DataFrame) {
    let df = windows_step_back_frame();
    let mut run = WindowsRun::new(windows_step_back_config(), df.schema()).unwrap();
    let _ = run.feed(&df.slice(0, 6), None).unwrap();
    assert!(run.total_skipped_step_back() >= 2);
    assert!(run.held() > 0);
    (run.save_bytes().unwrap(), df.slice(6, 4))
}

pub(super) fn windows_step_back_skip_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut run = WindowsRun::load_bytes(bytes, windows_step_back_config(), input.schema())?;
    let again = run.save_bytes()?;
    let fed = run.feed(input, None).map_err(|e| e.to_string())?;
    let finished = run.finish().map_err(|e| e.to_string())?;
    let notices = df!("notice" => run.take_notices()).map_err(|e| e.to_string())?;
    Ok((
        again,
        vec![
            ("feed".into(), fed),
            ("finish".into(), finished),
            ("notices".into(), notices),
        ],
    ))
}

fn merge_step_back_cfg() -> MergeConfig {
    MergeConfig {
        clock: "t".into(),
        source: "source".into(),
        by: Some("g".into()),
        restart: Some(rheon_polars::Restart::Float(10.0)),
        restart_text: Some("10".into()),
        skip_step_back: true,
        carry: vec!["x".into()],
    }
}

fn merge_step_back_streams() -> Vec<(&'static str, DataFrame)> {
    let a = df!(
        "t" => [1.0, 5.0, 3.0, 6.0, 8.0, 7.0, 10.0],
        "g" => ["a", "a", "a", "b", "a", "a", "b"],
        "x" => [10.0, 20.0, 99.0, 1.0, 30.0, 88.0, 2.0],
    )
    .unwrap();
    let b = df!(
        "t" => [2.0, 4.0, 7.0, 6.0, 9.0, 11.0],
        "g" => ["a", "b", "b", "b", "a", "b"],
        "x" => [100.0, 200.0, 300.0, 999.0, 400.0, 500.0],
    )
    .unwrap();
    vec![("a", a), ("b", b)]
}

fn merge_step_back_of(
    streams: &[(&'static str, DataFrame)],
    state: Option<&[u8]>,
) -> Result<Merge, String> {
    let names = streams.iter().map(|(n, _)| (*n).to_owned()).collect();
    let schemas = streams
        .iter()
        .map(|(_, d)| d.schema().as_ref().clone())
        .collect();
    let cfg = merge_step_back_cfg();
    match state {
        None => Merge::new(names, schemas, cfg).map_err(|e| e.to_string()),
        Some(bytes) => Merge::load_bytes(bytes, names, schemas, cfg).map_err(|e| e.to_string()),
    }
}

pub(super) fn merge_step_back_skip_first() -> (Vec<u8>, DataFrame) {
    let streams = merge_step_back_streams();
    let cut = [4usize, 3];
    let mut m = merge_step_back_of(&streams, None).unwrap();
    let first: Vec<DataFrame> = streams
        .iter()
        .zip(cut)
        .map(|((_, d), k)| d.head(Some(k)))
        .collect();
    merge_drive(&mut m, &first, None, true).unwrap();
    assert!(m.total_skipped_step_back() >= 1);
    let rest: Vec<DataFrame> = streams
        .iter()
        .zip(cut)
        .map(|((_, d), k)| d.slice(k as i64, d.height() - k))
        .collect();
    (m.save_bytes().unwrap(), merge_join(&rest))
}

pub(super) fn merge_step_back_skip_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let streams = merge_step_back_streams();
    let mut m = merge_step_back_of(&streams, Some(bytes))?;
    let again = m.save_bytes()?;
    let parts = merge_split(&streams, input)?;
    let out = merge_drive(&mut m, &parts, None, false)?;
    let notices = df!("notice" => m.take_notices()).map_err(|e| e.to_string())?;
    Ok((
        again,
        vec![("merged".into(), out), ("notices".into(), notices)],
    ))
}

fn refresh_step_back_frame() -> DataFrame {
    df!(
        "g" => ["g1", "g1", "g1", "g2", "g1", "g2", "g1", "g1", "g2", "g2", "g1", "g2"],
        "s" => ["a",  "b",  "a",  "a",  "b",  "b",  "a",  "b",  "a",  "b",  "a",  "a"],
        "t" => [1.0,  2.0,  1.5,  1.0,  3.0,  2.5,  2.0,  4.0,  2.0,  5.0,  6.0,  7.0],
        "v" => [1.0,  2.0,  99.0, 10.0, 3.0,  20.0, 88.0, 4.0,  77.0, 30.0, 5.0,  40.0],
    )
    .unwrap()
}

fn refresh_step_back_cols() -> RefreshCols<'static> {
    RefreshCols {
        series: "s",
        clock: "t",
        value: "v",
        group: Some("g"),
        keep: &[],
    }
}

pub(super) fn refresh_time_step_back_skip_first() -> (Vec<u8>, DataFrame) {
    let df = refresh_step_back_frame();
    let cols = refresh_step_back_cols();
    let mut rt = RefreshTime::new(vec!["a".into(), "b".into()], false)
        .unwrap()
        .with_skip_step_back(true);
    let _ = rt.feed(&df.slice(0, 6), &cols).unwrap();
    assert!(rt.total_skipped_step_back() >= 1);
    (rt.save_bytes().unwrap(), df.slice(6, 6))
}

pub(super) fn refresh_time_step_back_skip_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let cols = refresh_step_back_cols();
    let mut rt = RefreshTime::load_bytes_with_options(
        bytes,
        vec!["a".into(), "b".into()],
        false,
        true,
        false,
        true,
    )?;
    let again = rt.save_bytes()?;
    let out = rt.feed(input, &cols).map_err(|e| e.to_string())?;
    let notices = df!("notice" => rt.take_notices()).map_err(|e| e.to_string())?;
    Ok((
        again,
        vec![("grid".into(), out), ("notices".into(), notices)],
    ))
}

fn bars_step_back_cfg() -> BarsConfig {
    BarsConfig {
        kind: BarKind::Count,
        clock: "t".into(),
        threshold: BarThreshold::Value(3.0),
        quantity: None,
        group: Some("g".into()),
        weight: None,
        session: None,
        gap_cap: None,
        max_span: None,
        restart_after_step_back: None,
        on_step_back: Some("skip".into()),
        keep_partial: true,
        monotone: false,
        max_state_bytes: 256 * 1024,
        max_total_bytes: 1 << 30,
        aggregates: vec![
            AggSpec {
                kind: AggKind::First,
                column: Some("p".into()),
                name: "open".into(),
            },
            AggSpec {
                kind: AggKind::Last,
                column: Some("p".into()),
                name: "close".into(),
            },
            AggSpec {
                kind: AggKind::Len,
                column: None,
                name: "rows".into(),
            },
        ],
        call: Default::default(),
    }
}

fn bars_step_back_frame() -> DataFrame {
    df!(
        "t" => [1i64, 1, 3, 2, 5, 4, 6, 4, 8, 9, 7, 10],
        "g" => ["a",  "b", "a", "a", "a", "b", "b", "a", "a", "b", "b", "a"],
        "p" => [10.0, 20.0, 11.0, 99.0, 12.0, 21.0, 22.0, 88.0, 13.0, 23.0, 77.0, 14.0],
    )
    .unwrap()
}

pub(super) fn bars_step_back_skip_first() -> (Vec<u8>, DataFrame) {
    let df = bars_step_back_frame();
    let mut b = Bars::new(bars_step_back_cfg(), df.schema()).unwrap();
    let _ = b.feed(&df.slice(0, 6), None).unwrap();
    assert!(b.total_skipped_step_back() >= 1);
    (b.save_bytes().unwrap(), df.slice(6, 6))
}

pub(super) fn bars_step_back_skip_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut b = Bars::load_bytes(bytes, bars_step_back_cfg(), input.schema())?;
    let again = b.save_bytes()?;
    let out = b.feed(input, None).map_err(|e| e.to_string())?;
    let open = b.finish().map_err(|e| e.to_string())?;
    let notices = df!("notice" => b.take_step_back_notices()).map_err(|e| e.to_string())?;
    Ok((
        again,
        vec![
            ("bars".into(), out),
            ("open".into(), open),
            ("notices".into(), notices),
        ],
    ))
}

/// Hard rule 5: every new persisted form for `on_step_back = "skip"` is held
/// non-empty in its frozen fixture.
#[test]
fn the_step_back_skip_fixtures_freeze_their_forms() {
    if regenerating() {
        return;
    }
    let check_leaves = |name: &str| {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("missing fixture {name}"));
        let v = rmpv::decode::read_value(&mut unhex(f.state).as_slice()).unwrap();
        let mut all = Vec::new();
        leaves(&v, "", &mut all);
        all
    };

    let bank = check_leaves("bank_step_back_skip");
    assert!(
        bank.iter()
            .any(|(p, v)| p.ends_with(".on_step_back") && v.as_str() == Some("skip")),
        "bank_step_back_skip missing on_step_back"
    );
    assert!(
        bank.iter()
            .any(|(p, v)| p.ends_with(".clock_backwards") && v.as_u64().is_some_and(|n| n > 0)),
        "bank_step_back_skip missing non-zero clock_backwards"
    );

    let win = check_leaves("with_windows_step_back_skip");
    assert!(
        win.iter()
            .any(|(p, v)| p.ends_with(".config.on_step_back") && v.as_str() == Some("skip")),
        "with_windows_step_back_skip missing config.on_step_back"
    );
    assert!(
        win.iter()
            .any(|(p, v)| p.ends_with(".skipped_step_back") && v.as_u64().is_some_and(|n| n > 0)),
        "with_windows_step_back_skip missing non-zero skipped_step_back"
    );
    assert!(
        win.iter().any(|(p, _)| p.starts_with(".core.stale_seqs.")),
        "with_windows_step_back_skip missing non-empty stale_seqs"
    );

    let merge = check_leaves("merge_step_back_skip");
    assert!(
        merge
            .iter()
            .any(|(p, v)| p == ".skip_step_back" && v.as_bool() == Some(true)),
        "merge_step_back_skip missing skip_step_back"
    );
    assert!(
        merge.iter().any(|(p, v)| {
            p.starts_with(".skipped_step_back.") && v.as_u64().is_some_and(|n| n > 0)
        }),
        "merge_step_back_skip missing non-empty skipped_step_back"
    );

    let rt = check_leaves("refresh_time_step_back_skip");
    assert!(
        rt.iter()
            .any(|(p, v)| p == ".skip_step_back" && v.as_bool() == Some(true)),
        "refresh_time_step_back_skip missing skip_step_back"
    );
    assert!(
        rt.iter().any(|(p, v)| {
            p.starts_with(".skipped_step_back.") && v.as_u64().is_some_and(|n| n > 0)
        }),
        "refresh_time_step_back_skip missing non-empty skipped_step_back"
    );

    let bars = check_leaves("bars_step_back_skip");
    assert!(
        bars.iter()
            .any(|(p, v)| p == ".config.on_step_back" && v.as_str() == Some("skip")),
        "bars_step_back_skip missing config.on_step_back"
    );
    assert!(
        bars.iter().any(|(p, v)| {
            p.starts_with(".skipped_step_back.") && v.as_u64().is_some_and(|n| n > 0)
        }),
        "bars_step_back_skip missing non-empty skipped_step_back"
    );
}
