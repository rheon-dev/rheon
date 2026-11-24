//! Records the x86 CPU features this build was compiled to need, in
//! `python/rheon/_build_flags.py`, for the import-time CPU check
//! (`python/rheon/_cpu_check.py`).
//!
//! The wheels of `rheon` for x86-64 are built with
//! `-C target-feature=+avx2,+fma,...` (release.yml), so the compiler may
//! emit instructions an older CPU does not have, and that CPU would die on
//! an illegal instruction. The check runs in Python before the extension
//! module is loaded, and it needs to know what to look for. Reading the
//! features here, from the flags cargo hands rustc, makes the record the
//! flags themselves: a build cannot name one set of features and compile
//! for another. Polars records its own runtime wheels' features the same way
//! (`py-polars/runtime/polars-runtime-32/build.rs`).
//!
//! A build with no `target-feature` flag, and any build that is not for x86,
//! records the empty string, which is no check. The file is not committed:
//! it is written here, and a missing one is written again (the
//! `rerun-if-changed` on it below).
//!
//! Standard library only, so a test compiles this file with `rustc` and runs
//! it (tests/test_cpu_check.py).

use std::env;
use std::fs;
use std::path::PathBuf;

/// The x86-64 psABI levels' features in rustc's names (the levels of the
/// System V AMD64 ABI supplement; `lahf_lm` has no rustc feature), each
/// level's beyond the one below it.
const V2: &[&str] = &["cmpxchg16b", "popcnt", "sse3", "sse4.1", "sse4.2", "ssse3"];
const V3: &[&str] = &[
    "avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "lzcnt", "movbe", "xsave",
];
const V4: &[&str] = &["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"];

/// The features a `-C target-cpu=...` value needs: a psABI level's, and
/// none for another CPU (`native` is the building machine's, unknown here).
fn cpu_features(cpu: &str) -> Vec<String> {
    let levels: &[&[&str]] = match cpu {
        "x86-64-v2" => &[V2],
        "x86-64-v3" => &[V2, V3],
        "x86-64-v4" => &[V2, V3, V4],
        _ => &[],
    };
    levels
        .iter()
        .flat_map(|l| l.iter())
        .map(|f| f.to_string())
        .collect()
}

/// The features the flags in a `CARGO_ENCODED_RUSTFLAGS` value (the flags
/// joined by `0x1f`) make the build need, in order: the last `target-cpu`'s
/// psABI level, then each `target-feature` item in turn, `+name` adding one
/// and `-name` taking it away, as rustc reads them (review 2026-11-23,
/// PYO3-2: a `-name` after a `+name` was ignored, and a `target-cpu` level
/// recorded nothing). `crt-static`, a linking choice and no CPU feature, is
/// left out.
fn required_features(encoded: &str) -> Vec<String> {
    let flags: Vec<&str> = encoded.split('\u{1f}').filter(|f| !f.is_empty()).collect();
    let mut cpu: Vec<String> = Vec::new();
    let mut items: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < flags.len() {
        let flag = flags[i];
        // `-C name=...` (two words), `-Cname=...`, `--codegen name=...` and
        // `--codegen=name=...`.
        let value = if flag == "-C" || flag == "--codegen" {
            i += 1;
            flags.get(i).copied()
        } else if let Some(rest) = flag.strip_prefix("--codegen=") {
            Some(rest)
        } else {
            flag.strip_prefix("-C")
        };
        if let Some(list) = value.and_then(|v| v.strip_prefix("target-feature=")) {
            items.extend(list.split(','));
        } else if let Some(name) = value.and_then(|v| v.strip_prefix("target-cpu=")) {
            cpu = cpu_features(name);
        }
        i += 1;
    }
    let mut out = cpu;
    for item in items {
        let (on, name) = match item.split_at_checked(1) {
            Some(("+", name)) => (true, name),
            Some(("-", name)) => (false, name),
            _ => continue,
        };
        if name.is_empty() || name == "crt-static" {
            continue;
        }
        out.retain(|f| f != name);
        if on {
            out.push(name.to_string());
        }
    }
    out
}

/// The text of `_build_flags.py` for these features.
fn render(features: &[String]) -> String {
    let list: Vec<String> = features.iter().map(|f| format!("+{f}")).collect();
    format!(
        "# Written by crates/rheon-py/build.rs; not committed.\n\
         # The x86 CPU features this build was compiled to need, as rustc was told,\n\
         # empty for a build with none (the CPU check then does nothing).\n\
         BUILD_FEATURE_FLAGS = \"{}\"\n",
        list.join(",")
    )
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // crates/rheon-py/ -> the workspace root, where the Python source sits
    // (also in an sdist, which keeps the layout).
    let target = manifest_dir
        .join("..")
        .join("..")
        .join("python")
        .join("rheon")
        .join("_build_flags.py");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    // A file that is missing counts as changed, so a checkout that never had
    // one, or whose file was deleted, writes it even from a cached build.
    println!("cargo:rerun-if-changed={}", target.display());

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = if arch == "x86_64" || arch == "x86" {
        required_features(&env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default())
    } else {
        Vec::new()
    };
    let text = render(&features);

    // The Python source folder is the one place maturin reads from. A layout
    // without it (the crate built on its own) records nothing, which is no
    // check, and is no reason to fail the build.
    let Some(dir) = target.parent().filter(|d| d.is_dir()) else {
        return;
    };
    // Only when it differs, so a build that records the same flags leaves the
    // file's time alone and cargo does not run this again for it.
    if !fs::read_to_string(&target).is_ok_and(|old| old == text) {
        fs::write(dir.join("_build_flags.py"), text)
            .unwrap_or_else(|e| panic!("could not write {}: {e}", target.display()));
    }
}
