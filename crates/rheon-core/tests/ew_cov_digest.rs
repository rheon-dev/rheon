//! `ew_cov` and `ew_class` without shrinkage are the models they were
//! before shrinkage existed, to the bit: every output, every `n_eff` and
//! every state byte, over fixed seeded streams through each of their
//! optional parts (docs/PLAN.md task 511). A hash of all of them, held to the hash `origin/main` at 13f5ffe4
//! gave for this same file before the shrinkage landed.
//!
//! The configurations are read from JSON, which omits every field added
//! since, so the file compiles and runs unchanged on the commit before.
//!
//! Two hashes. The portable one reads no libm and no faer: `Lam` decay at
//! steps of 0 and 1 (`Decay::factor` is `lam` and 1 there by
//! construction), normals as sums of twelve uniforms, and no statistic that
//! factorizes; it is held on every platform. The whole one adds the
//! half-life decay (`exp2`), Box-Muller normals (`ln`, `cos`), and the
//! statistics read through faer's factorizations (`partial_corr`, `mahal`,
//! the components), whose last bits follow the math library and the CPU's
//! paths (docs/TESTING.md, "The math library's last bits"): it is held
//! where it was measured, Linux on x86-64. `ew_class`'s scores read `ln`
//! and `exp`, so its portable hash is of its `n_eff` and state bytes
//! alone, and its whole one adds the posteriors.

use rheon_core::{EwClass, EwClassCfg, EwCovCfg, EwCovModel, OnlineModel, State};

/// FNV-1a, 64 bits: a hash with no dependency and no platform in it.
struct Fnv(u64);

impl Fnv {
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= u64::from(x);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn float(&mut self, v: f64) {
        self.bytes(&v.to_bits().to_le_bytes());
    }
}

/// xorshift64*, uniform on `[0, 1)`.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller: `ln` and `cos`, libm's.
    fn normal(&mut self) -> f64 {
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// Irwin-Hall: twelve uniforms less 6, no libm.
    fn normal12(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// The portable configurations: no libm, no factorization.
fn portable() -> Vec<(usize, String)> {
    vec![
        (
            4,
            r#"{"n_features": 4, "decay": {"Lam": 0.95}, "stats": ["mean", "var", "std",
            "cov", "corr"], "min_weight": 0.0}"#
                .to_string(),
        ),
        (
            4,
            r#"{"n_features": 4, "decay": {"Lam": 0.9}, "stats": ["var", "cov", "corr"],
            "min_weight": 2.0, "bias": true}"#
                .to_string(),
        ),
        (
            3,
            r#"{"n_features": 3, "decay": {"Lam": 0.96}, "stats": ["mean", "var", "cov",
            "corr"], "min_weight": 0.0, "window": 30.0, "window_every": 2.0}"#
                .to_string(),
        ),
        (
            5,
            r#"{"n_features": 5, "decay": {"Lam": 0.93}, "stats": ["var", "std", "corr"],
            "min_weight": 0.0, "window": 18.0, "max_rows_between_snapshots": 3}"#
                .to_string(),
        ),
        (
            3,
            r#"{"n_features": 3, "decay": {"Lam": 0.94}, "stats": ["corr", "lag_corr"],
            "min_weight": 0.0, "lags": [1, 3]}"#
                .to_string(),
        ),
        (
            20,
            r#"{"n_features": 20, "decay": {"Lam": 0.98}, "stats": ["mean", "var", "corr"],
            "min_weight": 0.0}"#
                .to_string(),
        ),
        (
            2,
            r#"{"n_features": 2, "decay": {"Lam": 0.9}, "stats": [], "min_weight": 0.0}"#
                .to_string(),
        ),
    ]
}

/// The whole configurations: each optional part of `ew_cov` at least once.
fn whole() -> Vec<(usize, String)> {
    let all = r#"["mean", "var", "std", "cov", "corr", "partial_corr", "mahal"]"#;
    vec![
        (
            4,
            format!(
                r#"{{"n_features": 4, "decay": {{"Halflife": 20.0}}, "stats": {all},
                "min_weight": 0.0, "precision_prior": 1e-6}}"#
            ),
        ),
        (
            4,
            format!(
                r#"{{"n_features": 4, "decay": {{"Halflife": 15.0}}, "stats": {all},
                "min_weight": 2.0, "bias": true, "precision_prior": 1e-3, "pca": 2,
                "pca_every": 3.0, "max_rows_between_pca": 5, "mahal_quantiles": [0.5, 0.9]}}"#
            ),
        ),
        (
            3,
            r#"{"n_features": 3, "decay": {"Halflife": 25.0}, "stats": ["mean", "var", "cov",
            "corr", "mahal"], "min_weight": 0.0, "precision_prior": 1e-4, "pca": 1,
            "window": 30.0, "window_every": 2.0}"#
                .to_string(),
        ),
        (
            5,
            r#"{"n_features": 5, "decay": {"Halflife": 10.0}, "stats": ["var", "corr",
            "partial_corr"], "min_weight": 0.0, "precision_prior": 1e-2, "window": 18.0,
            "max_rows_between_snapshots": 3}"#
                .to_string(),
        ),
        (
            3,
            r#"{"n_features": 3, "decay": {"Halflife": 12.0}, "stats": ["corr", "lag_corr"],
            "min_weight": 0.0, "lags": [1, 3]}"#
                .to_string(),
        ),
        (
            20,
            r#"{"n_features": 20, "decay": {"Halflife": 50.0}, "stats": ["mean", "var",
            "corr"], "min_weight": 0.0}"#
                .to_string(),
        ),
        (
            2,
            r#"{"n_features": 2, "decay": {"Halflife": 8.0}, "stats": [], "min_weight": 0.0}"#
                .to_string(),
        ),
    ]
}

fn digest(cfgs: Vec<(usize, String)>, libm: bool) -> u64 {
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    for (seed, (k, json)) in cfgs.into_iter().enumerate() {
        let cfg: EwCovCfg = serde_json::from_str(&json).unwrap();
        let mut m = EwCovModel::new(cfg).unwrap();
        let mut g = Rng(0x9e37_79b9_7f4a_7c15 ^ (seed as u64 + 1));
        for r in 0..400 {
            let x: Vec<f64> = (0..k)
                .map(|i| {
                    let z = if libm { g.normal() } else { g.normal12() };
                    3.0 * i as f64 + (1.0 + 0.3 * i as f64) * z
                })
                .collect();
            // A zero-weight first row, rows of weight 0 between, unequal
            // weights, and gaps of a half, one and three clock units (of
            // none and one where no libm is read).
            let w = if r == 0 || r % 9 == 4 {
                0.0
            } else {
                0.5 + g.uniform()
            };
            let dt = if libm {
                [1.0, 0.5, 3.0][r % 3]
            } else {
                [1.0, 0.0, 1.0][r % 3]
            };
            let step = m.step(&x, &[], dt, w);
            for v in &step.pred {
                h.float(*v);
            }
            h.float(step.n_eff);
            if r % 100 == 99 {
                let state = m.state();
                h.bytes(&rmp_serde::to_vec_named(&state).unwrap());
                // Saved and loaded halfway, as a bank would.
                if r == 199 {
                    let bytes = rmp_serde::to_vec_named(&state).unwrap();
                    let back: State = rmp_serde::from_slice(&bytes).unwrap();
                    m = EwCovModel::restore(&back).unwrap();
                }
            }
        }
    }
    h.0
}

#[test]
fn without_shrinkage_every_output_and_state_byte_is_as_before() {
    assert_eq!(
        format!("{:016x}", digest(portable(), false)),
        "1a6a1da1414d7507",
        "ew_cov without shrinkage moved"
    );
}

#[test]
#[cfg_attr(
    not(all(target_os = "linux", target_arch = "x86_64")),
    ignore = "its libm and faer bits were measured on Linux x86-64; the portable digest runs here"
)]
fn without_shrinkage_every_factorized_output_is_as_before() {
    assert_eq!(
        format!("{:016x}", digest(whole(), true)),
        "3c332a8fc97cddc6",
        "ew_cov without shrinkage moved"
    );
}

/// `ew_class` in each shape, with and without a window: the label every
/// third row of each class, a null label now and then.
fn classifiers() -> Vec<String> {
    let mut out = Vec::new();
    for shape in ["full", "tied", "diag"] {
        for window in ["", r#", "window": 25.0, "window_every": 2.0"#] {
            out.push(format!(
                r#"{{"n_features": 3, "n_classes": 3, "decay": {{"Lam": 0.95}},
                "min_weight": 0.0, "covariance": "{shape}", "precision_prior": 0.01{window}}}"#
            ));
        }
    }
    out
}

fn classifier_digest(libm: bool) -> u64 {
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    for (seed, json) in classifiers().into_iter().enumerate() {
        let cfg: EwClassCfg = serde_json::from_str(&json).unwrap();
        let mut m = EwClass::new(cfg).unwrap();
        let mut g = Rng(0x51_7cc1_b727_220a ^ (seed as u64 + 1));
        for r in 0..300 {
            let c = r % 3;
            let x: Vec<f64> = (0..3)
                .map(|i| {
                    let z = if libm { g.normal() } else { g.normal12() };
                    c as f64 * (1.0 + i as f64) + z
                })
                .collect();
            let y = if r % 11 == 5 { None } else { Some(c as f64) };
            let w = if r == 0 || r % 9 == 4 {
                0.0
            } else {
                0.5 + g.uniform()
            };
            let dt = if libm {
                [1.0, 0.5, 3.0][r % 3]
            } else {
                [1.0, 0.0, 1.0][r % 3]
            };
            let step = m.step(&x, &[y], dt, w);
            if libm {
                for v in &step.pred {
                    h.float(*v);
                }
            }
            h.float(step.n_eff);
            if r % 100 == 99 {
                let state = m.state();
                h.bytes(&rmp_serde::to_vec_named(&state).unwrap());
                if r == 199 {
                    let bytes = rmp_serde::to_vec_named(&state).unwrap();
                    let back: State = rmp_serde::from_slice(&bytes).unwrap();
                    m = EwClass::restore(&back).unwrap();
                }
            }
        }
    }
    h.0
}

#[test]
fn without_shrinkage_every_classifier_state_byte_is_as_before() {
    assert_eq!(
        format!("{:016x}", classifier_digest(false)),
        "d8a441ad73bdcfa4",
        "ew_class without shrinkage moved"
    );
}

#[test]
#[cfg_attr(
    not(all(target_os = "linux", target_arch = "x86_64")),
    ignore = "its libm bits were measured on Linux x86-64; the portable digest runs here"
)]
fn without_shrinkage_every_classifier_output_is_as_before() {
    assert_eq!(
        format!("{:016x}", classifier_digest(true)),
        "9a95ecef92176e3c",
        "ew_class without shrinkage moved"
    );
}
