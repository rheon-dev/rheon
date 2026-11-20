//! Runner-level tests (docs/ENHANCEMENTS.md E32): the same numbers whichever
//! format the rows come in or go out in, frames from a plan or from an
//! iterator, files or callbacks out, and the edges -- empty streams, explicit
//! formats, a run that fails after an earlier one succeeded.

use std::path::{Path, PathBuf};

use polars::prelude::*;
use rheon_polars::{
    Bank, Format, Input, Output, RunConfig, RunOptions, RunStats, run, run_config, run_config_on,
};

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("rheon-runner-test-{}-{name}", std::process::id()));
    p
}

/// Deterministic two-group stream.
fn stream(n: usize) -> DataFrame {
    let mut s = 77u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let (mut group, mut t, mut x0, mut x1, mut y) = (vec![], vec![], vec![], vec![], vec![]);
    let mut clocks = [0.0f64, 0.0];
    for i in 0..n {
        let g = i % 2;
        group.push(format!("g{g}"));
        clocks[g] += 1.0 + lcg().abs();
        t.push(clocks[g]);
        let (a, b) = (lcg(), lcg());
        x0.push(a);
        x1.push(b);
        y.push(if g == 0 { 2.0 * a - b } else { -a } + 0.01 * lcg());
    }
    df!("group" => group, "t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

/// `df` written to `path` in `format` with polars' own writer.
fn write(path: &Path, format: Format, df: &DataFrame) {
    let mut df = df.clone();
    let file = std::fs::File::create(path).unwrap();
    match format {
        Format::Parquet => ParquetWriter::new(file).finish(&mut df).map(|_| ()),
        Format::Ipc => IpcWriter::new(file).finish(&mut df),
        Format::Csv => CsvWriter::new(file).finish(&mut df),
        Format::Ndjson => JsonWriter::new(file)
            .with_json_format(JsonFormat::JsonLines)
            .finish(&mut df),
    }
    .unwrap();
}

/// `path` read back with the runner's own scan.
fn read(path: &Path, format: Format) -> DataFrame {
    format.scan(path).unwrap().collect().unwrap()
}

fn names(df: &DataFrame) -> Vec<&str> {
    df.get_column_names()
        .into_iter()
        .map(|s| s.as_str())
        .collect()
}

/// The extension each format is told from.
fn ext(format: Format) -> &'static str {
    match format {
        Format::Parquet => "parquet",
        Format::Ipc => "arrow",
        Format::Csv => "csv",
        Format::Ndjson => "jsonl",
    }
}

/// `coef` on every row: `coef_every = 0`.
fn spec() -> rheon_polars::Spec {
    spec_with_cadence(r#""coef_every": 0"#)
}

/// The spec with `cadence`, the `coef` schedule's `"key": value` pair: the
/// clock is in seconds, so a cadence in rows is `max_rows_between_coefs`.
fn spec_with_cadence(cadence: &str) -> rheon_polars::Spec {
    serde_json::from_str(&format!(
        r#"{{
            "name": "ridge",
            "model": {{"type": "ewridge", "ridge": 1e-6, "max_rows_between_solves": 1}},
            "targets": ["y"],
            "features": ["x0", "x1"],
            "clock": "t",
            "half_life": 50.0,
            "gap_cap": 10.0,
            "group": "group",
            "min_weight": 5.0,
            {cadence}
        }}"#
    ))
    .unwrap()
}

fn config(input: &Path, output: &Path) -> RunConfig {
    RunConfig {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        input_format: None,
        output_format: None,
        output_layout: Default::default(),
        chunk_size: 64,
        threads: None,
        load_state: None,
        skip_learned: false,
        save_state: None,
        keep_columns: vec![],
        predict: false,
        closed_groups: None,
        specs: vec![spec()],
    }
}

/// `validate` on a config whose spec is a model with no target and no
/// `targets` -- the dict the bank fills from `features[0]` (E53) -- without
/// `fill_defaults` first. It checked each spec before the bank could fill
/// it, and refused (review 2026-09-12, S25).
#[test]
fn validate_accepts_a_spec_the_bank_fills() {
    let mut cfg = config(Path::new("in.parquet"), Path::new("out.parquet"));
    cfg.specs = vec![
        serde_json::from_str(
            r#"{"name": "c", "model": {"type": "ew_cov"}, "features": ["x0", "x1"],
                "half_life": 10.0}"#,
        )
        .unwrap(),
    ];
    cfg.validate().expect("the bank fills the targets");
}

fn no_progress(_: RunStats) -> PolarsResult<()> {
    Ok(())
}

/// `pred_y` from an output frame: the struct field, or the flattened CSV
/// column.
fn preds(df: &DataFrame) -> Vec<Option<f64>> {
    let s = match df.column("ridge") {
        Ok(c) => c.struct_().unwrap().field_by_name("pred_y").unwrap(),
        Err(_) => df
            .column("ridge.pred_y")
            .unwrap()
            .as_materialized_series()
            .clone(),
    };
    s.f64().unwrap().iter().collect()
}

/// What the bank says on its own, the oracle for every path through the
/// runner.
fn bank_preds(df: &DataFrame) -> Vec<Option<f64>> {
    let mut bank = Bank::new(vec![spec()]).unwrap();
    let cols = bank.fit_predict(df).unwrap();
    preds(&DataFrame::new(df.height(), cols).unwrap())
}

fn cleanup(paths: &[PathBuf]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn every_input_format_gives_the_banks_numbers() {
    let df = stream(500);
    let want = bank_preds(&df);
    let mut files = vec![];
    for format in Format::ALL {
        let input = tmp(&format!("in-{}.{}", format.name(), ext(format)));
        let output = tmp(&format!("in-{}-out.parquet", format.name()));
        write(&input, format, &df);
        let stats = run_config(&config(&input, &output), no_progress).unwrap();
        assert_eq!((stats.rows, stats.chunks), (500, 8), "{format:?}");
        let got = read(&output, Format::Parquet);
        assert_eq!(preds(&got), want, "{format:?} input");
        // The input columns come through as they were.
        assert_eq!(got.drop("ridge").unwrap(), df, "{format:?} input");
        files.extend([input, output]);
    }
    cleanup(&files);
}

#[test]
fn every_output_format_carries_the_banks_columns() {
    let df = stream(500);
    let want = bank_preds(&df);
    let input = tmp("out-in.parquet");
    write(&input, Format::Parquet, &df);
    let mut files = vec![input.clone()];
    for format in Format::ALL {
        let output = tmp(&format!("out-{}.{}", format.name(), ext(format)));
        run_config(&config(&input, &output), no_progress).unwrap();
        let got = read(&output, format);
        assert_eq!(got.height(), 500, "{format:?}");
        assert_eq!(preds(&got), want, "{format:?} output");
        match format {
            // CSV has no nested values: the struct is flattened to
            // `<spec>.<field>` columns and the list field is a JSON string.
            Format::Csv => {
                assert_eq!(
                    names(&got),
                    [
                        "group",
                        "t",
                        "x0",
                        "x1",
                        "y",
                        "ridge.pred_y",
                        "ridge.resid_y",
                        "ridge.weight_sum",
                        "ridge.settled_frac",
                        "ridge.withheld_reason",
                        "ridge.coef",
                        "ridge.support_coef"
                    ]
                );
                let coef = got.column("ridge.coef").unwrap();
                assert_eq!(coef.dtype(), &DataType::String);
                let last = coef.str().unwrap().last().unwrap();
                assert!(last.starts_with('[') && last.ends_with(']'), "{last}");
                // Intercept and two features.
                assert_eq!(last.split(',').count(), 3, "{last}");
            }
            _ => {
                let ridge = got.column("ridge").unwrap();
                assert!(matches!(ridge.dtype(), DataType::Struct(_)), "{format:?}");
                let coef = ridge.struct_().unwrap().field_by_name("coef").unwrap();
                // A list in every format, as the bank writes it: a mostly
                // null `coef` stores nothing on the rows that carry none.
                assert_eq!(
                    coef.dtype(),
                    &DataType::List(Box::new(DataType::Float64)),
                    "{format:?}"
                );
            }
        }
        files.push(output);
    }
    cleanup(&files);
}

#[test]
fn csv_lists_read_back_as_lists() {
    let df = stream(200);
    let input = tmp("csv-list-in.parquet");
    let output = tmp("csv-list-out.csv");
    write(&input, Format::Parquet, &df);
    // A snapshot every 7 rows, so the rows between are null and must come
    // back null (an empty CSV field) rather than as an empty list.
    let mut cfg = config(&input, &output);
    cfg.specs = vec![spec_with_cadence(r#""max_rows_between_coefs": 7"#)];
    run_config(&cfg, no_progress).unwrap();
    let got = read(&output, Format::Csv);
    let nulls = got.column("ridge.coef").unwrap().null_count();
    assert!(nulls > 100 && nulls < 200, "{nulls}");
    // The documented way back is `str.json_decode(pl.List(pl.Float64))`,
    // polars' JSON parser (this test is the guarantee since tests/test_runner.py
    // went with the Python runner in task 83); here,
    // the same parse by hand with the standard library's correctly rounded
    // `f64::from_str`. (serde_json's default float parsing is best-effort
    // and lands an ulp off on some of these.)
    let coef = got.column("ridge.coef").unwrap();
    let parsed: Vec<Option<Vec<f64>>> = coef
        .str()
        .unwrap()
        .iter()
        .map(|s| {
            s.map(|s| {
                s.trim_matches(['[', ']'])
                    .split(',')
                    .map(|x| x.parse::<f64>().unwrap())
                    .collect()
            })
        })
        .collect();
    // The oracle sees the runner's chunks, though under a cadence no row
    // of `coef` moves with them (docs/PLAN.md task 178).
    let mut bank = Bank::new(vec![spec_with_cadence(r#""max_rows_between_coefs": 7"#)]).unwrap();
    let mut want: Vec<Option<Vec<f64>>> = vec![];
    for start in (0..df.height()).step_by(64) {
        let cols = bank.fit_predict(&df.slice(start as i64, 64)).unwrap();
        let coef = cols[0].struct_().unwrap().field_by_name("coef").unwrap();
        want.extend(
            coef.list()
                .unwrap()
                .amortized_iter()
                .map(|s| s.map(|s| s.as_ref().f64().unwrap().iter().flatten().collect())),
        );
    }
    assert_eq!(parsed, want);
    cleanup(&[input, output]);
}

#[test]
fn frames_from_an_iterator_equal_the_plan() {
    let df = stream(700);
    let output_plan = tmp("frames-plan.parquet");
    let output_iter = tmp("frames-iter.parquet");
    let cfg = config(Path::new(""), &output_plan);
    run_config_on(&cfg, Input::Lazy(df.clone().lazy()), no_progress).unwrap();
    // Uneven frames, and a size `chunk_size` says nothing about.
    let frames = (0..700)
        .step_by(150)
        .map(move |i| Ok(df.slice(i as i64, 150)));
    let cfg = config(Path::new(""), &output_iter);
    let stats = run_config_on(
        &cfg,
        Input::Batches {
            frames: Box::new(frames),
            schema: stream(0).schema().as_ref().clone(),
        },
        no_progress,
    )
    .unwrap();
    assert_eq!((stats.rows, stats.chunks), (700, 5));
    assert_eq!(
        read(&output_iter, Format::Parquet),
        read(&output_plan, Format::Parquet)
    );
    cleanup(&[output_plan, output_iter]);
}

#[test]
fn keep_columns_applies_to_frames_too() {
    let df = stream(100)
        .with_column(Column::new("extra".into(), vec![1i32; 100]))
        .unwrap()
        .clone();
    let output = tmp("keep-frames.parquet");
    let mut cfg = config(Path::new(""), &output);
    cfg.keep_columns = ["group", "t", "x0", "x1", "y"].map(String::from).to_vec();
    let frames = std::iter::once(Ok(df.clone()));
    run_config_on(
        &cfg,
        Input::Batches {
            frames: Box::new(frames),
            schema: df.schema().as_ref().clone(),
        },
        no_progress,
    )
    .unwrap();
    let got = read(&output, Format::Parquet);
    assert_eq!(names(&got), ["group", "t", "x0", "x1", "y", "ridge"]);
    cleanup(&[output]);
}

#[test]
fn frames_to_a_callback_equal_the_file() {
    let df = stream(300);
    let output = tmp("callback.parquet");
    let cfg = config(Path::new(""), &output);
    run_config_on(&cfg, Input::Lazy(df.clone().lazy()), no_progress).unwrap();
    let from_file = read(&output, Format::Parquet);

    let mut bank = Bank::new(vec![spec()]).unwrap();
    let mut got: Vec<DataFrame> = vec![];
    let mut sink = |frame: DataFrame| {
        got.push(frame);
        Ok(())
    };
    let opts = RunOptions {
        chunk_size: 64,
        predict: false,
        learn_only: false,
    };
    let stats = run(
        &mut bank,
        Input::Lazy(df.lazy()),
        Output::Batches(&mut sink),
        opts,
        no_progress,
    )
    .unwrap();
    assert_eq!((stats.rows, stats.chunks), (300, 5));
    assert_eq!(got.len(), 5);
    let whole = got
        .into_iter()
        .reduce(|a, b| a.vstack(&b).unwrap())
        .unwrap();
    assert_eq!(whole, from_file);
    cleanup(&[output]);
}

#[test]
fn an_empty_stream_writes_an_empty_output_in_every_format() {
    let empty = stream(0);
    let input = tmp("empty-in.parquet");
    write(&input, Format::Parquet, &empty);
    let mut files = vec![input.clone()];
    for format in Format::ALL {
        let output = tmp(&format!("empty-out.{}", ext(format)));
        let stats = run_config(&config(&input, &output), no_progress).unwrap();
        assert_eq!((stats.rows, stats.chunks), (0, 0));
        match format {
            // No rows, no header, no footer: an empty file.
            Format::Ndjson => assert_eq!(std::fs::metadata(&output).unwrap().len(), 0),
            _ => {
                let got = read(&output, format);
                assert_eq!(got.height(), 0, "{format:?}");
                let last = names(&got).last().unwrap().to_string();
                assert!(last.starts_with("ridge"), "{format:?}: {last}");
            }
        }
        files.push(output);
    }
    // And from frames that turn out empty.
    let output = tmp("empty-frames.parquet");
    let cfg = config(Path::new(""), &output);
    run_config_on(
        &cfg,
        Input::Batches {
            frames: Box::new(std::iter::empty()),
            schema: empty.schema().as_ref().clone(),
        },
        no_progress,
    )
    .unwrap();
    assert_eq!(
        names(&read(&output, Format::Parquet)),
        ["group", "t", "x0", "x1", "y", "ridge"]
    );
    files.push(output);
    cleanup(&files);
}

#[test]
fn an_explicit_format_overrides_the_extension() {
    let df = stream(100);
    let input = tmp("explicit-in.dat");
    let output = tmp("explicit-out.bin");
    write(&input, Format::Ipc, &df);
    let mut cfg = config(&input, &output);
    assert!(
        run_config(&cfg, no_progress)
            .unwrap_err()
            .to_string()
            .contains("cannot tell its format"),
        "an unknown extension names the fix"
    );
    cfg.input_format = Some(Format::Ipc);
    cfg.output_format = Some(Format::Ndjson);
    run_config(&cfg, no_progress).unwrap();
    assert_eq!(preds(&read(&output, Format::Ndjson)), bank_preds(&df));
    cleanup(&[input, output]);
}

#[test]
fn a_failed_run_leaves_the_previous_output_where_it_was() {
    let df = stream(200);
    let output = tmp("failed-run.parquet");
    let cfg = config(Path::new(""), &output);
    run_config_on(&cfg, Input::Lazy(df.clone().lazy()), no_progress).unwrap();
    let before = std::fs::read(&output).unwrap();

    // The source fails halfway.
    let frames = vec![
        Ok(df.slice(0, 100)),
        Err(polars_err!(ComputeError: "the source went away")),
    ];
    let err = run_config_on(
        &cfg,
        Input::Batches {
            frames: Box::new(frames.into_iter()),
            schema: df.schema().as_ref().clone(),
        },
        no_progress,
    )
    .unwrap_err();
    assert!(err.to_string().contains("the source went away"), "{err}");
    assert_eq!(std::fs::read(&output).unwrap(), before);

    // And so does a progress callback that gives up.
    let err = run_config_on(&cfg, Input::Lazy(df.lazy()), |s| {
        if s.chunks == 2 {
            polars_bail!(ComputeError: "stop here");
        }
        Ok(())
    })
    .unwrap_err();
    assert!(err.to_string().contains("stop here"), "{err}");
    assert_eq!(std::fs::read(&output).unwrap(), before);
    // No temporary is left behind.
    let dir = output.parent().unwrap();
    let stem = output.file_name().unwrap().to_str().unwrap();
    let leftovers: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(stem) && n != stem)
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    cleanup(&[output]);
}

/// Twelve groups of five rows in key order: under `monotone` a group closes
/// when the next key arrives, so two chunks of ten close three groups.
fn closing_stream() -> DataFrame {
    let n = 60;
    let g: Vec<String> = (0..n).map(|i| format!("k{:02}", i / 5)).collect();
    let x0: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| ((i * 5) % 13) as f64).collect();
    df!("g" => g, "x0" => x0, "y" => y).unwrap()
}

fn closing_spec() -> rheon_polars::Spec {
    serde_json::from_str(
        r#"{"name": "c", "model": {"type": "ew_cov"}, "features": ["x0", "y"],
            "half_life": 40.0, "group": "g", "group_close": "monotone"}"#,
    )
    .unwrap()
}

/// A run that fails still publishes the closed groups it drained before it
/// failed: a drained row has left the bank, so the sidecar is the only place
/// it is -- as `fit_predict_batches` writes on a `break` or an error (the
/// second review's F2). The output keeps its rule: a product of the whole
/// run, published by a run that reached its end.
#[test]
fn a_failed_run_publishes_the_closed_groups_it_drained() {
    use rheon_polars::Bank;
    let df = closing_stream();
    let output = tmp("failed-closed.parquet");
    let closed = tmp("failed-closed-groups.parquet");
    let mut cfg = config(Path::new(""), &output);
    cfg.specs = vec![closing_spec()];
    cfg.closed_groups = Some(closed.clone());
    let frames = vec![
        Ok(df.slice(0, 10)),
        Ok(df.slice(10, 10)),
        Err(polars_err!(ComputeError: "the source went away")),
    ];
    let err = run_config_on(
        &cfg,
        Input::Batches {
            frames: Box::new(frames.into_iter()),
            schema: df.schema().as_ref().clone(),
        },
        no_progress,
    )
    .unwrap_err();
    assert!(err.to_string().contains("the source went away"), "{err}");
    assert!(!output.exists(), "the output is a product of the whole run");
    // What those two chunks closed, a bank fed the same rows says.
    let mut bank = Bank::new(vec![closing_spec()]).unwrap();
    bank.fit_predict(&df.slice(0, 20)).unwrap();
    let want = bank.closed_groups(None, true).unwrap();
    assert_eq!(want.height(), 3, "three groups closed before the failure");
    assert!(closed.exists(), "the sidecar holds what was drained");
    let got = read(&closed, Format::Parquet);
    assert!(got.equals_missing(&want), "{got}\nvs\n{want}");
    cleanup(&[closed]);
}

#[test]
fn progress_reports_every_chunk() {
    let df = stream(250);
    let output = tmp("progress.parquet");
    let cfg = config(Path::new(""), &output);
    let mut seen = vec![];
    run_config_on(&cfg, Input::Lazy(df.lazy()), |s| {
        seen.push((s.rows, s.chunks));
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, [(64, 1), (128, 2), (192, 3), (250, 4)]);
    cleanup(&[output]);
}

/// A glob input is read in the lexicographic order of its paths, which is
/// polars' own order for an expanded glob (review round 4, SF5): `part-10`
/// before `part-2`, where the numbers say otherwise. Pinned because a
/// deployment depends on it -- name the parts so the two orders agree, by
/// padding the numbers -- and so a change of it shows here. Here the parts
/// hold rows 0..200 and 200..400 of one stream, and the output, which
/// keeps the input's columns in the order they were read, starts with the
/// rows of `part-10`. With a clock, the same files are refused as a step
/// back at the first row of `part-2`.
#[test]
fn a_glob_input_is_read_in_the_lexicographic_order_of_its_paths() {
    let dir = tmp("glob-order");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let df = stream(400);
    let id = Column::new("id".into(), (0..400i64).collect::<Vec<_>>());
    let df = df.hstack(&[id]).unwrap();
    // Numeric order: part-2 first, then part-10. Lexicographic: the reverse.
    write(
        &dir.join("part-2.parquet"),
        Format::Parquet,
        &df.slice(0, 200),
    );
    write(
        &dir.join("part-10.parquet"),
        Format::Parquet,
        &df.slice(200, 200),
    );
    let output = dir.join("out.parquet");
    let mut cfg = config(&dir.join("part-*.parquet"), &output);
    // No clock: the order is read off the output, not refused.
    cfg.specs[0].clock = None;
    cfg.specs[0].gap_cap = None;
    run_config(&cfg, no_progress).unwrap();
    let ids: Vec<i64> = read(&output, Format::Parquet)
        .column("id")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let want: Vec<i64> = (200..400).chain(0..200).collect();
    assert_eq!(ids, want, "part-10's rows, then part-2's");
    // A clock steps back at part-2's first row, the 201st read.
    let mut clocked = config(&dir.join("part-*.parquet"), &output);
    clocked.specs[0].group = None;
    let mut ordered = df.clone();
    ordered
        .with_column(Column::new(
            "t".into(),
            (0..400).map(f64::from).collect::<Vec<f64>>(),
        ))
        .unwrap();
    write(
        &dir.join("part-2.parquet"),
        Format::Parquet,
        &ordered.slice(0, 200),
    );
    write(
        &dir.join("part-10.parquet"),
        Format::Parquet,
        &ordered.slice(200, 200),
    );
    let err = run_config(&clocked, no_progress).unwrap_err().to_string();
    assert!(
        err.contains("steps back") && err.contains("row 200"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- Task 252: runner paths no test reached. ----

/// What a config cannot ask for together, each refused before a row is
/// read, in words that say which key to drop.
#[test]
fn validate_refuses_a_config_that_asks_for_what_cannot_be() {
    let base = config(Path::new("in.parquet"), Path::new("out.parquet"));
    let refused = |cfg: RunConfig, want: &str| {
        let e = cfg.validate().unwrap_err();
        assert!(e.contains(want), "wanted {want:?} in: {e}");
    };
    let mut c = base.clone();
    c.specs = vec![closing_spec()];
    c.predict = true;
    c.load_state = Some("state.bin".into());
    c.closed_groups = Some("closed.parquet".into());
    refused(c, "predict = true learns nothing, so no group ever closes");
    let mut c = base.clone();
    c.closed_groups = Some("closed.parquet".into());
    refused(c, "closed_groups names a file, but no spec closes groups");
    let mut c = base.clone();
    c.predict = true;
    refused(c, "predict = true needs load_state");
    let mut c = base.clone();
    c.predict = true;
    c.load_state = Some("state.bin".into());
    c.save_state = Some("again.bin".into());
    refused(
        c,
        "predict = true learns nothing, so save_state has nothing to save",
    );
    let mut c = base.clone();
    c.input = PathBuf::new();
    assert_eq!(
        c.input_format().unwrap_err(),
        "config: the run has no input. Set input in the config, or pass --input."
    );
}

/// A config that leaves out `chunk_size` reads at the default, as the
/// options a library run starts from do.
#[test]
fn a_config_without_chunk_size_reads_at_the_default() {
    let cfg: RunConfig = toml::from_str(
        r#"input = "in.parquet"
output = "out.parquet"
[[specs]]
name = "m"
targets = ["y"]
features = ["x0"]
half_life = 10.0
model = { type = "ewridge" }
"#,
    )
    .unwrap();
    assert_eq!(cfg.chunk_size, rheon_polars::DEFAULT_CHUNK_SIZE);
    let opts = RunOptions::default();
    assert_eq!(opts.chunk_size, rheon_polars::DEFAULT_CHUNK_SIZE);
    assert!(!opts.predict && !opts.learn_only);
}

/// An output in a directory that is not there is refused before the bank
/// learns a row, naming the file and the directory.
#[test]
fn an_output_in_a_directory_that_is_not_there_is_refused_first() {
    let dir = tmp("no-such-dir");
    let _ = std::fs::remove_dir_all(&dir);
    let output = dir.join("out.parquet");
    let cfg = config(Path::new(""), &output);
    let e = run_config_on(&cfg, Input::Lazy(stream(10).lazy()), no_progress).unwrap_err();
    let msg = e.to_string();
    assert!(
        msg.contains(&format!("writing output {}", output.display()))
            && msg.contains(&format!("{} is not a directory", dir.display())),
        "{msg}"
    );
    assert!(matches!(e, PolarsError::IO { .. }), "{e:?}");
}

/// The closed groups sidecar: a run that closes nothing writes it empty with
/// its schema; a run whose first chunk closes nothing and a later one does
/// writes what was closed; and as CSV, a Gram model's with its column names
/// as JSON text, and another model's written whole.
#[test]
fn the_closed_groups_file_holds_what_closed_in_any_format() {
    let df = closing_stream();
    let output = tmp("closed-out.parquet");
    // One group: nothing ever closes.
    let one = df.slice(0, 5);
    let closed = tmp("closed-none.parquet");
    let mut cfg = config(Path::new(""), &output);
    cfg.specs = vec![closing_spec()];
    cfg.closed_groups = Some(closed.clone());
    run_config_on(&cfg, Input::Lazy(one.lazy()), no_progress).unwrap();
    let got = read(&closed, Format::Parquet);
    assert_eq!(got.height(), 0);
    let mut bank = rheon_polars::Bank::new(vec![closing_spec()]).unwrap();
    bank.fit_predict(&df).unwrap();
    let want = bank.closed_groups(None, true).unwrap();
    assert_eq!(names(&got), names(&want), "the schema of a closing run");
    // Chunks of 3: the first holds one key and closes nothing.
    cfg.chunk_size = 3;
    run_config_on(&cfg, Input::Lazy(df.clone().lazy()), no_progress).unwrap();
    let got = read(&closed, Format::Parquet);
    assert_eq!(got.height(), 11, "every group but the last");
    assert!(got.equals_missing(&want), "{got}\nvs\n{want}");
    // As CSV: a closed frame has no struct to unnest. A Gram's column names
    // are a list of strings, written as JSON text as a numeric list is, and
    // read back as the list it was.
    let csv = tmp("closed.csv");
    cfg.closed_groups = Some(csv.clone());
    run_config_on(&cfg, Input::Lazy(df.clone().lazy()), no_progress).unwrap();
    let got = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(csv.clone()))
        .unwrap()
        .finish()
        .unwrap();
    assert_eq!(got.height(), want.height());
    let lists = want.column("columns").unwrap().list().unwrap();
    let texts = got.column("columns").unwrap().str().unwrap();
    for (list, text) in lists.amortized_iter().zip(texts.iter()) {
        let names: Vec<String> = list
            .unwrap()
            .as_ref()
            .str()
            .unwrap()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect();
        let read: Vec<String> = serde_json::from_str(text.unwrap()).unwrap();
        assert_eq!(read, names);
    }
    // A model that keeps no Gram closes rows CSV carries whole.
    cfg.specs = vec![
        serde_json::from_str(
            r#"{"name": "s", "model": {"type": "sgd"}, "targets": ["y"], "features": ["x0"],
                "half_life": 40.0, "group": "g", "group_close": "monotone"}"#,
        )
        .unwrap(),
    ];
    run_config_on(&cfg, Input::Lazy(df.clone().lazy()), no_progress).unwrap();
    let text = std::fs::read_to_string(&csv).unwrap();
    let mut bank = rheon_polars::Bank::new(cfg.specs.clone()).unwrap();
    bank.fit_predict(&df).unwrap();
    let want = bank.closed_groups(None, true).unwrap();
    let header: Vec<&str> = text.lines().next().unwrap().split(',').collect();
    assert_eq!(header, names(&want));
    assert_eq!(text.lines().count(), 1 + want.height());
    cleanup(&[output, closed, csv]);
}

/// A run with a closed-groups file that fails before anything closed
/// publishes neither file.
#[test]
fn a_run_that_fails_before_anything_closed_publishes_no_sidecar() {
    let df = closing_stream();
    let output = tmp("fail-early-out.parquet");
    let closed = tmp("fail-early-closed.parquet");
    let _ = std::fs::remove_file(&closed);
    let mut cfg = config(Path::new(""), &output);
    cfg.specs = vec![closing_spec()];
    cfg.closed_groups = Some(closed.clone());
    let frames = vec![
        Ok(df.slice(0, 3)),
        Err(polars_err!(ComputeError: "the source went away")),
    ];
    let err = run_config_on(
        &cfg,
        Input::Batches {
            frames: Box::new(frames.into_iter()),
            schema: df.schema().as_ref().clone(),
        },
        no_progress,
    )
    .unwrap_err();
    assert!(err.to_string().contains("the source went away"), "{err}");
    assert!(!output.exists() && !closed.exists());
}

/// A dry run of a resumed config that skips what it learned takes the
/// learned positions before its first row: a bank whose specs read no clock
/// has none, and is refused there, as the run refuses it.
#[test]
fn a_dry_run_that_skips_what_was_learned_takes_the_positions_first() {
    let input = tmp("dry-skip-in.parquet");
    let output = tmp("dry-skip-out.parquet");
    let state = tmp("dry-skip-state.bin");
    write(&input, Format::Parquet, &stream(50));
    let mut cfg = config(&input, &output);
    cfg.save_state = Some(state.clone());
    run_config(&cfg, no_progress).unwrap();
    let mut resumed = config(&input, &output);
    resumed.load_state = Some(state.clone());
    resumed.skip_learned = true;
    resumed.dry_run().unwrap();
    // Without a clock there is no position to resume from.
    let mut rows = config(&input, &output);
    rows.specs[0].clock = None;
    rows.specs[0].gap_cap = None;
    rows.save_state = Some(state.clone());
    run_config(&rows, no_progress).unwrap();
    let mut resumed = rows.clone();
    resumed.save_state = None;
    resumed.load_state = Some(state.clone());
    resumed.skip_learned = true;
    let e = resumed.dry_run().unwrap_err().to_string();
    assert!(
        e.contains(
            "config: skip_learned = true (--skip-learned) needs a spec that reads a clock, but no \
             spec does, so the bank has no position to resume from"
        ),
        "{e}"
    );
    cleanup(&[input, output, state]);
}

/// The run's closing lines on readiness pass over a spec that writes no
/// row -- an `audit` keeps its counts in the state, and emits no
/// `withheld_reason` to count -- and speak of the spec that does.
#[test]
fn readiness_lines_pass_over_a_spec_that_withholds_nothing() {
    let df = stream(200);
    let output = tmp("readiness-audit.parquet");
    let mut cfg = config(Path::new(""), &output);
    let audit: rheon_polars::Spec = serde_json::from_str(
        r#"{"name": "a", "model": {"type": "audit"}, "targets": ["x0"],
            "features": ["x0", "x1"]}"#,
    )
    .unwrap();
    let mut slow = spec();
    slow.name = "slow".into();
    slow.min_weight = Some(rheon_polars::FloatOrList::Float(rheon_polars::Num(1e9)));
    cfg.specs = vec![audit, slow];
    let (stats, lines) =
        rheon_polars::run_config_on_reported(&cfg, Input::Lazy(df.lazy()), no_progress).unwrap();
    assert_eq!(stats.rows, 200);
    assert!(!lines.is_empty(), "the slow spec withheld every row");
    assert!(lines.iter().all(|l| !l.contains("\"a\"")), "{lines:?}");
    assert!(lines.iter().any(|l| l.contains("slow")), "{lines:?}");
    cleanup(&[output]);
}

/// A writer that cannot create its file -- a directory the run may read but
/// not write -- ends the run with the writer's own error, naming the file,
/// not the run's "the writer stopped"; for the output and for the closed
/// groups alike, and nothing is published.
///
/// Unix only: a directory is made unwritable by its mode bits, which
/// Windows does not read (its read-only attribute on a directory does not
/// stop a file being created in it). Skipped, saying so, where the process
/// may write past the mode bits, as root does.
#[cfg(unix)]
#[test]
fn a_writer_that_cannot_create_its_file_ends_the_run_with_its_own_error() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tmp("read-only-dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    // A process that may write past a directory's mode -- root, which holds
    // `CAP_DAC_OVERRIDE` -- creates the file anyway, so there is no refusal
    // to test: the probe tries the very write the runner would make, and
    // skips with the reason where it succeeds.
    let probe = dir.join("probe");
    if std::fs::File::create(&probe).is_ok() {
        eprintln!(
            "skipped: this process can create a file in a 0555 directory \
             (running as root, or with CAP_DAC_OVERRIDE), so no writer is refused"
        );
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let df = closing_stream();
    let writable = tmp("read-only-out.parquet");
    for which in ["output", "closed groups"] {
        let (output, closed) = if which == "output" {
            (dir.join("out.parquet"), tmp("read-only-closed.parquet"))
        } else {
            (writable.clone(), dir.join("closed.parquet"))
        };
        let mut cfg = config(Path::new(""), &output);
        cfg.specs = vec![closing_spec()];
        cfg.closed_groups = Some(closed.clone());
        let e = run_config_on(&cfg, Input::Lazy(df.clone().lazy()), no_progress).unwrap_err();
        let bad = if which == "output" { &output } else { &closed };
        let msg = e.to_string();
        assert!(
            matches!(e, PolarsError::IO { .. }) && msg.contains(&format!("{}", bad.display())),
            "{which}: {msg}"
        );
        assert!(!msg.contains("the writer stopped"), "{which}: {msg}");
        assert!(!output.exists() || which != "output", "{which}");
        cleanup(&[writable.clone(), tmp("read-only-closed.parquet")]);
    }
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// `stream(n)` with a Categorical column `venue` whose second category,
/// `"B"`, first comes at row `late`.
fn with_late_category(n: usize, late: usize) -> DataFrame {
    let venue: Vec<&str> = (0..n).map(|i| if i < late { "A" } else { "B" }).collect();
    let venue = Series::new("venue".into(), venue)
        .cast(&DataType::from_categories(Categories::global()))
        .unwrap();
    let mut df = stream(n);
    df.with_column(venue.into_column()).unwrap();
    df
}

/// Review P2: an IPC output of a run over a Categorical column whose
/// category first comes in a later chunk. Arrow's IPC file holds one
/// dictionary a field for every batch, so the run failed at that chunk --
/// at 10 rows a chunk -- where the same input as one chunk was written: the
/// output depended on `chunk_size`. Now it is written at any chunk size,
/// the same file each time.
#[test]
fn an_ipc_output_does_not_depend_on_when_a_category_first_comes() {
    let df = with_late_category(100, 90);
    let input = tmp("late-category-in.parquet");
    write(&input, Format::Parquet, &df);
    let mut got = Vec::new();
    for chunk_size in [1000, 10, 7] {
        let output = tmp(&format!("late-category-{chunk_size}.arrow"));
        let mut cfg = config(&input, &output);
        cfg.chunk_size = chunk_size;
        let stats = run_config(&cfg, no_progress)
            .unwrap_or_else(|e| panic!("chunk_size {chunk_size}: {e}"));
        assert_eq!(stats.rows, 100);
        got.push(read(&output, Format::Ipc));
        cleanup(&[output]);
    }
    for g in &got[1..] {
        assert!(g.equals_missing(&got[0]), "{g}\nvs\n{}", got[0]);
    }
    let venue: Vec<Option<String>> = got[0]
        .column("venue")
        .unwrap()
        .cast(&DataType::String)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|v| v.map(str::to_string))
        .collect();
    assert_eq!(venue[89].as_deref(), Some("A"));
    assert_eq!(venue[90].as_deref(), Some("B"));
    assert_eq!(preds(&got[0]), bank_preds(&df));
    cleanup(&[input]);
}

/// `stream(n)` with `day`, a Datetime in microseconds zoned to New York:
/// midnight there on 1 January 2024 for the first half of the rows and on
/// the 2nd for the second, which is 05:00 UTC.
fn with_zoned_day(n: usize) -> DataFrame {
    const US_PER_DAY: i64 = 86_400 * 1_000_000;
    // 2024-01-01T05:00:00Z in microseconds.
    let first = 1_704_085_200 * 1_000_000;
    let day: Vec<i64> = (0..n)
        .map(|i| first + if i < n / 2 { 0 } else { US_PER_DAY })
        .collect();
    let zone = TimeZone::opt_try_new(Some("America/New_York")).unwrap();
    // Built on the integers: a cast to a zone drops it in a build without
    // polars' time zones.
    let day = Int64Chunked::from_vec("day".into(), day)
        .into_datetime(TimeUnit::Microseconds, zone)
        .into_series();
    assert!(matches!(day.dtype(), DataType::Datetime(_, Some(_))));
    let mut df = stream(n);
    df.with_column(day.into_column()).unwrap();
    df
}

/// Review P3: a zoned Datetime column failed as a group column (the
/// closing lines cast it to text as a naive one), and in CSV and NDJSON
/// output, which have no form for a zone in this build -- the NDJSON
/// writer panicked. Python takes all three, and so does the run now: the
/// group's key is its instant in UTC, as the bank's is, and the text
/// formats write each instant in UTC, saying so.
#[test]
fn a_zoned_datetime_column_runs_as_a_group_and_into_every_format() {
    let df = with_zoned_day(40);
    let input = tmp("zoned-in.parquet");
    write(&input, Format::Parquet, &df);
    // As the group column, with every row withheld so the closing lines
    // name each group.
    let output = tmp("zoned-group.parquet");
    let mut cfg = config(&input, &output);
    cfg.specs[0].group = Some("day".into());
    // Each of `stream`'s two groups has a clock of its own.
    cfg.specs[0].clock = None;
    cfg.specs[0].gap_cap = None;
    cfg.specs[0].min_weight = Some(rheon_polars::FloatOrList::Float(rheon_polars::Num(1e9)));
    let (stats, lines) = rheon_polars::run_config_reported(&cfg, no_progress).unwrap();
    assert_eq!(stats.rows, 40);
    assert!(
        lines
            .iter()
            .any(|l| l.contains("2 groups whose last row was withheld")),
        "{lines:?}"
    );
    cleanup(&[output]);
    // Into each text format, the instant written in UTC.
    for format in [Format::Csv, Format::Ndjson] {
        let output = tmp(&format!("zoned-out.{}", ext(format)));
        let cfg = config(&input, &output);
        run_config(&cfg, no_progress).unwrap_or_else(|e| panic!("{format:?}: {e}"));
        let text = std::fs::read_to_string(&output).unwrap();
        // Polars' text for each format, the instant in UTC.
        let (first, second) = match format {
            Format::Csv => (
                "2024-01-01T05:00:00.000000+0000",
                "2024-01-02T05:00:00.000000+0000",
            ),
            _ => ("2024-01-01T05:00:00+00:00", "2024-01-02T05:00:00+00:00"),
        };
        assert!(
            text.contains(first) && text.contains(second),
            "{format:?}: {}",
            &text[..text.len().min(400)]
        );
        cleanup(&[output]);
    }
    // A clock in the output's struct, zoned, is written the same way.
    let mut zoned_clock = df.clone();
    let zone = TimeZone::opt_try_new(Some("America/New_York")).unwrap();
    let t = zoned_clock
        .column("t")
        .unwrap()
        .cast(&DataType::Int64)
        .unwrap()
        .as_materialized_series()
        .clone();
    let t = (t * 1_000_000i64)
        .i64()
        .unwrap()
        .clone()
        .into_datetime(TimeUnit::Microseconds, zone)
        .into_series();
    zoned_clock.with_column(t.into_column()).unwrap();
    let clocked = tmp("zoned-clock-in.parquet");
    write(&clocked, Format::Parquet, &zoned_clock);
    for format in [Format::Csv, Format::Ndjson] {
        let output = tmp(&format!("zoned-clock-out.{}", ext(format)));
        let mut cfg = config(&clocked, &output);
        cfg.specs = vec![
            serde_json::from_str(
                r#"{"name": "ridge", "model": {"type": "ewridge"}, "targets": ["y"],
                    "features": ["x0", "x1"], "clock": "t", "half_life": "50s",
                    "gap_cap": "10s", "group": "group", "emit_clocks": true}"#,
            )
            .unwrap(),
        ];
        run_config(&cfg, no_progress).unwrap_or_else(|e| panic!("{format:?}: {e}"));
        let text = std::fs::read_to_string(&output).unwrap();
        let offset = if format == Format::Csv {
            "+0000"
        } else {
            "+00:00"
        };
        assert!(text.contains(offset), "{format:?}: {}", &text[..400]);
        cleanup(&[output]);
    }
    cleanup(&[input, clocked]);
}

/// Review P13: a dry run did not lay out the output, so an input column
/// named as a flattened field of a spec -- `ridge.pred_y` -- passed it for
/// a flat output and a CSV, and the run then failed on the duplicate name.
#[test]
fn a_dry_run_lays_out_the_output() {
    let mut df = stream(20);
    let clash = df
        .column("y")
        .unwrap()
        .clone()
        .with_name("ridge.pred_y".into());
    df.with_column(clash).unwrap();
    let input = tmp("dry-layout-in.parquet");
    write(&input, Format::Parquet, &df);
    let parquet = tmp("dry-layout-out.parquet");
    let csv = tmp("dry-layout-out.csv");
    config(&input, &parquet)
        .dry_run()
        .expect("the struct layout keeps the spec's fields in its column");
    let mut flat = config(&input, &parquet);
    flat.output_layout = rheon_polars::OutputLayout::Flat;
    for (what, cfg) in [("flat", flat), ("csv", config(&input, &csv))] {
        let dry = cfg.dry_run().map(|_| ()).unwrap_err().to_string();
        let ran = run_config(&cfg, no_progress).unwrap_err().to_string();
        assert!(dry.contains("ridge.pred_y"), "{what}: {dry}");
        assert_eq!(dry, ran, "{what}: the dry run refuses as the run does");
    }
    cleanup(&[input, parquet, csv]);
}

/// Found beside review P3: this build's CSV writer has no form for a
/// `Date`, a `Time` or a naive `Datetime` either, so a CSV output of an
/// input with one was refused, where Python writes it. Each is written as
/// Polars writes it.
#[test]
fn a_csv_output_writes_dates_and_times_as_polars_does() {
    let mut df = stream(4);
    // 2024-01-01 in days, at midnight in milliseconds, and 01:02:03.000004.
    let day = Int32Chunked::from_vec("day".into(), vec![19_723; 4]).into_date();
    let at = Int64Chunked::from_vec("at".into(), vec![1_704_067_200_000; 4])
        .into_datetime(TimeUnit::Milliseconds, None);
    let time = Int64Chunked::from_vec("time".into(), vec![3_723_000_004_000; 4]).into_time();
    for s in [day.into_series(), at.into_series(), time.into_series()] {
        df.with_column(s.into_column()).unwrap();
    }
    let input = tmp("temporal-csv-in.parquet");
    let output = tmp("temporal-csv-out.csv");
    write(&input, Format::Parquet, &df);
    run_config(&config(&input, &output), no_progress).unwrap();
    let text = std::fs::read_to_string(&output).unwrap();
    let row = text.lines().nth(1).unwrap();
    assert!(
        row.contains(",2024-01-01,2024-01-01T00:00:00.000,01:02:03.000004000,"),
        "{row}"
    );
    cleanup(&[input, output]);
}

/// Review P4: two of a run's paths naming one file were taken, and the run
/// wrote one over the other -- the state over the output or the input, the
/// output over the state the next run resumes from. Each pair is refused,
/// however the path is spelled; `load_state` and `save_state` may be one.
#[test]
fn two_paths_naming_one_file_are_refused() {
    let dir = tmp("one-file-dir");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("in.parquet");
    write(&input, Format::Parquet, &closing_stream());
    // The same file through `..`, and a file that is not there yet.
    let respelled = dir
        .join("..")
        .join(dir.file_name().unwrap())
        .join("in.parquet");
    let other = |name: &str| dir.join(name);
    let base = || {
        let mut cfg = config(&input, &other("out.parquet"));
        cfg.specs = vec![closing_spec()];
        cfg
    };
    type Set = fn(&mut RunConfig, PathBuf);
    let roles: [(&str, Set); 5] = [
        ("input", |c, p| c.input = p),
        ("output", |c, p| c.output = p),
        ("load_state", |c, p| c.load_state = Some(p)),
        ("save_state", |c, p| c.save_state = Some(p)),
        ("closed_groups", |c, p| c.closed_groups = Some(p)),
    ];
    for (i, (a, set_a)) in roles.iter().enumerate() {
        for (b, set_b) in &roles[i + 1..] {
            let mut cfg = base();
            // A name with a format's extension serves every role.
            let shared = if *a == "input" {
                input.clone()
            } else {
                other("x.parquet")
            };
            let spelled = if *a == "input" {
                respelled.clone()
            } else {
                shared.clone()
            };
            set_a(&mut cfg, shared);
            set_b(&mut cfg, spelled);
            let got = cfg.validate();
            if (*a, *b) == ("load_state", "save_state") {
                assert!(got.is_ok(), "{a} and {b}: {got:?}");
                continue;
            }
            let e = got.unwrap_err();
            assert!(
                e.contains(&format!("config: {a} and {b} both name")),
                "{a} and {b}: {e}"
            );
        }
    }
    // Distinct paths pass.
    let mut cfg = base();
    cfg.save_state = Some(other("s.state"));
    cfg.closed_groups = Some(other("closed.parquet"));
    cfg.validate().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Review P9: a `save_state` in a directory the process may not write
/// passed the check before the run, which asked only whether the directory
/// was there, and the save failed after the output was published -- and a
/// dry run said the config was fine. Both refuse it before a row is read,
/// and the output is not written.
///
/// Unix only, and skipped where the process may write past the mode bits,
/// as `a_writer_that_cannot_create_its_file_ends_the_run_with_its_own_error`
/// is.
#[cfg(unix)]
#[test]
fn an_unwritable_state_directory_is_refused_before_the_run() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tmp("read-only-state-dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    let probe = dir.join("probe");
    if std::fs::File::create(&probe).is_ok() {
        eprintln!(
            "skipped: this process can create a file in a 0555 directory \
             (running as root, or with CAP_DAC_OVERRIDE), so no save is refused"
        );
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let input = tmp("read-only-state-in.parquet");
    let output = tmp("read-only-state-out.parquet");
    write(&input, Format::Parquet, &stream(30));
    let state = dir.join("s.state");
    let mut cfg = config(&input, &output);
    cfg.save_state = Some(state.clone());
    let dry = cfg.dry_run().unwrap_err();
    let ran = run_config(&cfg, no_progress).unwrap_err();
    for e in [&dry, &ran] {
        let msg = e.to_string();
        assert!(
            matches!(e, PolarsError::IO { .. })
                && msg.contains(&format!("saving state {}", state.display())),
            "{msg}"
        );
    }
    assert!(!output.exists(), "the output is not written");
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    cleanup(&[input, output]);
}
