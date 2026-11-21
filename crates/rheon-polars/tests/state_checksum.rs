//! A bank's and a window run's state file end with a checksum of their bytes
//! (schema 71, windows state 11; review P7), checked before anything is
//! decoded. Every byte of a saved file changed -- by its low bit, its high
//! bit, to 0, to 0xff and to 0x7f -- and every prefix, is refused, and none
//! panics or aborts. Before the checksum about one damage in fifty that
//! still decoded loaded in silence and changed the output, a few loaded into
//! a state that panicked at its next chunk, and a held row's Arrow IPC whose
//! length byte read 0x7f made polars-arrow ask for 3.6e16 bytes, which
//! aborts the process: a test reaching that case would not fail, it would
//! end the run, so this one finishing is part of what it holds.

use polars::prelude::*;
use rheon_polars::{Bank, Spec, WindowsConfig, WindowsRun};

/// The value a byte is changed to, each way.
fn changes(b: u8) -> impl Iterator<Item = u8> {
    [b ^ 0x01, b ^ 0x80, 0x00, 0xff, 0x7f]
        .into_iter()
        .filter(move |&c| c != b)
}

/// The checksum's entry: the key, its marker and the eight bytes.
const ENTRY: usize = 19;

fn stream(n: usize) -> DataFrame {
    let mut s = 485u64;
    let mut r = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let (mut t, mut g, mut x, mut mid, mut w) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut clock, mut level) = (0.0, 100.0);
    for i in 0..n {
        clock += 0.5 + r().abs();
        t.push(clock);
        g.push(if i % 3 == 0 { "a" } else { "b" });
        x.push(r());
        level += r();
        mid.push(level);
        w.push(if i % 7 == 3 { 0.0 } else { 1.0 });
    }
    df!("t" => t, "g" => g, "x" => x, "mid" => mid, "w" => w).unwrap()
}

/// A formula target, whose window core holds rows as Arrow IPC inside the
/// bank file, beside a Huber fit that emits its scale.
fn specs() -> Vec<Spec> {
    [
        r#"{"name": "fwd", "model": {"type": "ewridge"}, "features": ["x"], "group": "g",
            "targets": [{"name": "ahead", "formula": ["-", ["rewm_mean", ["col", "mid"],
                {"half_life": 5.0, "window_size": 10.0}], ["col", "mid"]]}],
            "clock": "t", "gap_cap": 100.0, "half_life": 50.0, "embargo": 10.0}"#,
        r#"{"name": "robust", "model": {"type": "huber", "emit_huber_scale": true}, "targets": ["mid"],
            "features": ["x"], "clock": "t", "gap_cap": 100.0, "half_life": 20.0, "weight": "w"}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

fn windows_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "ahead", "tree": ["rewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}]},
               {"name": "q", "tree": ["ewm_quantile", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0, "quantile": 0.9}]}
             ],
             "clock": "t", "gap_cap": 50.0, "group": "g"}"#,
    )
    .unwrap()
}

fn bank_bytes() -> Vec<u8> {
    let df = stream(40);
    let mut bank = Bank::new(specs()).unwrap();
    bank.fit_predict(&df).unwrap();
    bank.save_bytes().unwrap()
}

fn windows_input() -> DataFrame {
    stream(40).select(["t", "g", "mid"]).unwrap()
}

fn windows_bytes() -> Vec<u8> {
    let df = windows_input();
    let mut run = WindowsRun::new(windows_config(), df.schema()).unwrap();
    run.feed(&df, None).unwrap();
    assert!(run.held() > 0, "rows held for the windows ahead");
    run.save_bytes().unwrap()
}

/// Every change at every byte and every `prefix_step`-th prefix of `bytes`
/// refused by `load`, the changes before the checksum's entry by the
/// checksum's own words. A prefix holds no checksum, so its header is read,
/// a walk of the whole map: every prefix of a bank file is the extended
/// tier's.
fn every_damage_is_refused(
    bytes: &[u8],
    prefix_step: usize,
    load: impl Fn(&[u8]) -> Result<(), String>,
    says: &str,
) {
    load(bytes).unwrap_or_else(|e| panic!("the file as saved: {e}"));
    let mut tried = 0;
    for pos in 0..bytes.len() {
        for c in changes(bytes[pos]) {
            let mut b = bytes.to_vec();
            b[pos] = c;
            let err = load(&b)
                .err()
                .unwrap_or_else(|| panic!("byte {pos} set to {c:#04x} loaded"));
            if pos < bytes.len() - ENTRY {
                assert!(err.contains(says), "byte {pos} set to {c:#04x}: {err}");
            }
            tried += 1;
        }
    }
    assert!(tried >= 4 * bytes.len(), "{tried} changes tried");
    for n in (0..bytes.len()).step_by(prefix_step) {
        assert!(load(&bytes[..n]).is_err(), "cut at {n} of {}", bytes.len());
    }
}

fn bank_sweep(prefix_step: usize) {
    let bytes = bank_bytes();
    every_damage_is_refused(
        &bytes,
        prefix_step,
        |b| Bank::load_bytes(b, Some(&specs())).map(|_| ()),
        "state file: it is damaged (its bytes do not match the checksum it was saved with). \
         Load an undamaged copy of the state file, or rebuild it from its input.",
    );
}

#[test]
fn every_byte_of_a_bank_file_changed_is_refused() {
    bank_sweep(37);
}

#[test]
#[ignore = "extended: every prefix of a bank file as well (5 s)"]
fn every_byte_and_every_prefix_of_a_bank_file_is_refused() {
    bank_sweep(1);
}

#[test]
fn every_byte_of_a_window_runs_state_changed_is_refused() {
    let bytes = windows_bytes();
    let schema = windows_input().schema().clone();
    every_damage_is_refused(
        &bytes,
        1,
        |b| {
            WindowsRun::load_bytes(b, windows_config(), &schema)
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
        "with_windows: the state is damaged (its bytes do not match the checksum it was saved \
         with). Load an undamaged copy of the state file, or leave out load_state to start over.",
    );
}

/// The map `bytes` hold, without its `checksum` entry, and with its schema
/// field set to `schema` if given.
fn unsealed(bytes: &[u8], version_key: &str, version: Option<u64>) -> Vec<u8> {
    let mut v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    let rmpv::Value::Map(m) = &mut v else {
        panic!("a state file is a map")
    };
    assert_eq!(m.last().unwrap().0.as_str(), Some("checksum"), "last");
    m.pop();
    if let Some(version) = version {
        let (_, val) = m
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some(version_key))
            .unwrap();
        *val = rmpv::Value::from(version);
    }
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    out
}

/// A file of the sealed schema without its checksum is damaged; the same
/// file of the schema before, which wrote none, loads as one, and saves
/// the sealed file back.
#[test]
fn a_file_of_the_sealed_schema_without_its_checksum_is_damaged() {
    let bytes = bank_bytes();
    let err = Bank::load_bytes(&unsealed(&bytes, "", None), Some(&specs()))
        .err()
        .expect("refused");
    assert_eq!(
        err,
        "state file: it is damaged (the checksum it ends with is missing or cut). Load an \
         undamaged copy of the state file, or rebuild it from its input."
    );
    // The resolver's own core, version 11, is sealed in its own right, and
    // the bank's 70 loader takes the file whole, so only the bank's entry
    // goes.
    let old = unsealed(&bytes, "schema_version", Some(70));
    let bank = Bank::load_bytes(&old, Some(&specs())).unwrap();
    assert_eq!(bank.save_bytes().unwrap(), bytes);

    let w = windows_bytes();
    let schema = windows_input().schema().clone();
    let err = WindowsRun::load_bytes(&unsealed(&w, "", None), windows_config(), &schema)
        .err()
        .expect("refused")
        .to_string();
    assert_eq!(
        err,
        "with_windows: the state is damaged (the checksum it ends with is missing or cut). \
         Load an undamaged copy of the state file, or leave out load_state to start over."
    );
    let old = unsealed(&w, "version", Some(10));
    let run = WindowsRun::load_bytes(&old, windows_config(), &schema).unwrap();
    assert_eq!(run.save_bytes().unwrap(), w);
}

/// The JSON export is for reading, not loading: it carries no checksum.
#[test]
fn the_json_export_carries_no_checksum() {
    let df = stream(40);
    let mut bank = Bank::new(specs()).unwrap();
    bank.fit_predict(&df).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&bank.save_json_string(false).unwrap()).unwrap();
    assert!(json.get("checksum").is_none());
    assert_eq!(json["schema_version"], 71);
}

/// A file is hashed as it is written and the bytes in memory are sealed
/// after: the two are one file, byte for byte.
#[test]
fn a_saved_file_is_the_saved_bytes() {
    let df = stream(40);
    let mut bank = Bank::new(specs()).unwrap();
    bank.fit_predict(&df).unwrap();
    let dir = std::env::temp_dir().join(format!("rheon-checksum-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bank.state");
    bank.save(&path).unwrap();
    let on_disk = std::fs::read(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(on_disk, bank.save_bytes().unwrap());
}

/// A bank file of schema 70 holding a formula target's window core of
/// version 10, neither with a checksum, as the 70 build wrote it: it loads,
/// goes on to the bit of the sealed file loaded, and then saves that
/// bank's bytes.
#[test]
fn a_bank_of_70_with_a_core_of_10_goes_on_to_the_bit() {
    let df = stream(60);
    let mut bank = Bank::new(specs()).unwrap();
    bank.fit_predict(&df.slice(0, 40)).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let mut v =
        rmpv::decode::read_value(&mut &unsealed(&bytes, "schema_version", Some(70))[..]).unwrap();
    let rmpv::Value::Map(m) = &mut v else {
        panic!("a bank file is a map")
    };
    let (_, resolvers) = m
        .iter_mut()
        .find(|(k, _)| k.as_str() == Some("resolvers"))
        .expect("a formula target keeps its core");
    fn list(v: &mut rmpv::Value) -> &mut Vec<rmpv::Value> {
        match v {
            rmpv::Value::Array(a) => a,
            other => panic!("a list, not {other}"),
        }
    }
    let mut cores = 0;
    for spec in list(resolvers) {
        for group in list(&mut list(spec)[1]) {
            let held = &mut list(group)[1];
            let core: Vec<u8> = match held {
                rmpv::Value::Binary(b) => b.clone(),
                rmpv::Value::Array(a) => a.iter().map(|x| x.as_u64().unwrap() as u8).collect(),
                other => panic!("a core is bytes, not {other}"),
            };
            let old = unsealed(&core, "version", Some(10));
            *held = match held {
                rmpv::Value::Binary(_) => rmpv::Value::Binary(old),
                _ => rmpv::Value::Array(old.into_iter().map(rmpv::Value::from).collect()),
            };
            cores += 1;
        }
    }
    assert_eq!(cores, 2, "a core for each group");
    let mut old = Vec::new();
    rmpv::encode::write_value(&mut old, &v).unwrap();
    // A group's core is read at its next chunk, and kept as it came until
    // then, so it is after the chunk that the two save one file.
    let mut loaded = Bank::load_bytes(&old, Some(&specs())).unwrap();
    let tail = df.slice(40, 20);
    // The sealed file loaded and fed the same rows: a loaded core knows
    // its input from the load on, as the 70 bank's does.
    let mut bank = Bank::load_bytes(&bytes, Some(&specs())).unwrap();
    let want = bank.fit_predict(&tail).unwrap();
    let got = loaded.fit_predict(&tail).unwrap();
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert!(g.equals_missing(w), "{g:?} against {w:?}");
    }
    assert_eq!(loaded.save_bytes().unwrap(), bank.save_bytes().unwrap());
}
