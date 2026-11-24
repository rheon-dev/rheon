//! The 2026-11-23 review's `kalman` findings (PLAN 578f), each test first:
//! it failed on the build before the fix for the review's reason, or,
//! marked `should_panic`, holds a confirmed defect whose fix is the user's
//! call. A file of its own because `kalman.rs` stands near the source-size
//! cap (`tests/test_repo_hygiene.py`).

use super::*;

/// The plain form: `P` in the data's units with a given noise, every slot
/// sized at once, the intercept fitted.
fn plain_cfg(obs_var: f64, h: f64) -> KalmanCfg {
    KalmanCfg {
        horizons: Vec::new(),
        min_weight_holds_learning: false,
        noise_units: false,
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(h),
        half_life: vec![h],
        q: None,
        obs_var: Some(obs_var),
        p0: 4.0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    }
}

fn rows(n: usize, seed: u64) -> Vec<([f64; 2], f64)> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let x = [lcg(&mut s), 2.0 + lcg(&mut s)];
            (x, 1.0 + 3.0 * x[0] - x[1] + 0.1 * lcg(&mut s))
        })
        .collect()
}

/// `|a − b|` in units of the last place of the larger.
fn ulps_apart(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if scale == 0.0 {
        return 0.0;
    }
    (a - b).abs() / (scale * f64::EPSILON)
}

/// KAL-2: the factor's rank-one update rotates `(l_jj, x_j)` by their norm,
/// which squared both: below `1.5e-154` the squares underflowed and the
/// rotation was `0/0`, NaN; above `1.3e154` they overflowed and the
/// rotation `(0, 0)` wrote zeros over the factor's row, a finite factor no
/// check refuses, its variance gone. The update is the same at every scale
/// of two: at `2^±530` it is the update at 1 times that scale, within an
/// ulp or two of each entry.
#[test]
fn the_factors_rank_one_update_holds_at_every_scale() {
    // `L` lower triangular, row-major, and `x`.
    let l = [1.5, 0.0, 0.0, -0.5, 2.0, 0.0, 0.25, 0.75, 1.25];
    let x = [0.5, -1.0, 2.0];
    let mut unit_l = l;
    let mut unit_x = x;
    forms::factor_add(&mut unit_l, &mut unit_x);
    // `L'ᵀL' = LᵀL + x xᵀ` at scale 1, by the product.
    let k = 3;
    let before = forms::factor_product(&l, k);
    let after = forms::factor_product(&unit_l, k);
    for i in 0..k {
        for j in 0..k {
            let want = before[i * k + j] + x[i] * x[j];
            assert!((after[i * k + j] - want).abs() < 1e-14, "({i}, {j})");
        }
    }
    for e in [-530, 530] {
        let s = 2f64.powi(e);
        let mut sl = l.map(|v| v * s);
        let mut sx = x.map(|v| v * s);
        forms::factor_add(&mut sl, &mut sx);
        for (i, (got, want)) in sl.iter().zip(&unit_l).enumerate() {
            assert!(
                ulps_apart(got / s, *want) <= 4.0,
                "scale 2^{e}, entry {i}: {got:e} against {:e}",
                want * s
            );
        }
    }
}

/// KAL-3: a row of weight `1e100` against a noise of `1e-250` has `r = R /
/// w` of 0, an exact observation, which the covariance form takes: `P ←
/// P − P z zᵀ P / zᵀ P z`. The square root's update divided by `α_0 = r`
/// and wrote NaN into the factor, which the filter then threw away whole.
/// Both forms now give the same `P` after the row.
#[test]
fn an_exact_observation_is_the_same_in_both_forms() {
    let data = rows(40, 11);
    let run = |form: CovarianceForm| {
        let mut m = Kalman::new(plain_cfg(1e-250, 30.0))
            .unwrap()
            .with_covariance_form(form)
            .unwrap();
        for (i, (x, y)) in data.iter().enumerate() {
            let w = if i == 30 { 1e100 } else { 1.0 };
            m.step(x, &[Some(*y)], 1.0, w);
        }
        m
    };
    let (cov, root) = (
        run(CovarianceForm::Covariance),
        run(CovarianceForm::SquareRoot),
    );
    let (a, b) = (cov.cov(0).into_owned(), root.cov(0).into_owned());
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0 && b.iter().all(|v| v.is_finite()), "{b:?}");
    for (i, (p, q)) in a.iter().zip(&b).enumerate() {
        assert!(
            (p - q).abs() <= 1e-9 * scale,
            "entry {i}: {p:e} against {q:e}"
        );
    }
    for (p, q) in cov.beta[0].iter().zip(&root.beta[0]) {
        assert!((p - q).abs() <= 1e-9 * p.abs().max(1.0), "{p} {q}");
    }
}

/// KAL-4: a new filter's covariance moved to the square root takes each
/// diagonal entry's root; moved back, it takes the square again, so the
/// two moves leave the filter as it was, to the rounding of the root.
#[test]
fn a_form_given_and_given_back_leaves_the_covariance_as_it_was() {
    let m = Kalman::new(plain_cfg(0.5, 30.0)).unwrap();
    assert!(m.p[0].iter().any(|v| *v != 0.0 && *v != 1.0));
    let back = m
        .clone()
        .with_covariance_form(CovarianceForm::SquareRoot)
        .unwrap()
        .with_covariance_form(CovarianceForm::Covariance)
        .unwrap();
    // A root and its square round once each: within an ulp.
    for (got, want) in back.p[0].iter().zip(&m.p[0]) {
        assert!(ulps_apart(*got, *want) <= 1.0, "{got} against {want}");
    }
}

/// KAL-5: the noise a covariance has gone without, which the readouts add,
/// stays off a diffuse intercept still waiting for its first row, in the
/// plain form as in units of the noise: `add_process_noise` keeps it off
/// there, and `covariance_now` added it.
#[test]
fn a_diffuse_intercepts_readout_takes_no_noise_of_the_gap() {
    let mut cfg = plain_cfg(0.5, 30.0);
    cfg.noise_units = true;
    cfg.obs_var = None;
    let mut m = Kalman::new(cfg).unwrap();
    for (x, y) in rows(10, 3) {
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    // The intercept as a first row finds it: its row and column 0, the
    // clock since the last observation 5.
    let k = m.cfg.k_total();
    for i in 0..k {
        m.p[0][i] = 0.0;
        m.p[0][i * k] = 0.0;
    }
    m.elapsed[0] = 5.0;
    assert!(m.open::<false>(0) && !m.readiness_noise(0).is_nan());
    let now = m.covariance_now(0);
    assert_eq!(now[0], 0.0, "the open intercept took noise");
    assert!(
        now[k + 1] > m.p[0][k + 1],
        "the slopes take the gap's noise"
    );
}

/// KAL-7: a coefficient's variance read in the current coordinates, `u_0 +
/// Σ c_j u_j`, is `eᵀ P e` with `e = (1, c)`, and where a re-map was refused
/// `c` is past 1024 and the form cancels: `P` nearly singular along `e`
/// read below 0, which the error inflation `sqrt(1 + trace)` takes. A
/// rounding below 0 is 0, as the coefficients' own variance takes it. The
/// state is set by hand: the anchor `2000` of the feature's scale from its
/// mean, and `P = v vᵀ` with `v ⊥ e`, each scale of `v` tried until one
/// rounds below 0.
#[test]
fn the_intercepts_variance_in_the_current_coordinates_is_never_below_0() {
    let mut cfg = plain_cfg(0.5, 30.0);
    cfg.standardize = true;
    cfg.noise_units = true;
    cfg.obs_var = None;
    let mut m = Kalman::new(cfg).unwrap();
    let mut s = 5u64;
    for _ in 0..50 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(&x, &[Some(1.0 + 0.5 * x[1] + 0.1 * lcg(&mut s))], 1.0, 1.0);
    }
    let (mean, _, _) = m.stats.parts();
    m.anchor_hi[1] = mean[1] - 2000.0 * m.anchor_scale[1];
    let k = m.cfg.k_total();
    let live: Vec<bool> = (0..k).map(|i| m.is_live(i)).collect();
    let (mut a, mut c) = (vec![1.0; k], vec![0.0; k]);
    m.map_into(&live, &mut a, &mut c);
    assert!(c[1] > 1024.0, "c {c:?}");
    let mut below = 0;
    for t in 1..=200 {
        let t = 1.0 + t as f64 / 97.0;
        let v = [c[1] * t, -t, 0.0];
        for i in 0..k {
            for j in 0..k {
                m.p[0][i * k + j] = v[i] * v[j] + if i == j { 1e-300 } else { 0.0 };
            }
        }
        let read = m.current_diagonal(0)[0];
        assert!(read >= 0.0, "t {t}: the intercept's variance read {read:e}");
        below += usize::from(read == 0.0);
    }
    assert!(below > 0, "no scale rounded below 0");
}

/// KAL-6, claim 2: rows that observe no target still move the moments,
/// so the moments have no gap, but the floor on their history read the
/// clock since a target was observed, which runs on across them, and it
/// opened on 16 of 20 such rows at a half-life of 5. By the definition the
/// floor is shut on a row within a half-life of the last that moved the
/// moments: here every row, a clock step of 1 at a half-life of 5. So the
/// moments are the plain recursion's on every row, to the bit, an
/// `EwDiag` fed the same rows beside the filter the oracle. The run of
/// missing targets is long enough that the clock since a target was
/// observed passes the half-life on 16 of its rows, and one row in ten
/// weighs 1,000, more than the history's decayed weight, where an open
/// floor binds: read from that clock, the moments left the plain recursion
/// there.
#[test]
fn rows_with_no_target_leave_the_moments_floor_shut() {
    let mut cfg = plain_cfg(0.5, 5.0);
    cfg.standardize = true;
    cfg.noise_units = true;
    cfg.obs_var = None;
    let mut m = Kalman::new(cfg).unwrap();
    let mut plain = EwDiag::new(3);
    let lam = m.cfg.decay.factor(1.0);
    let mut s = 9u64;
    let (mut opened, mut past) = (0, 0);
    for i in 0..60 {
        let x = [lcg(&mut s), lcg(&mut s)];
        // Every row learns the features; the target is missing on rows 20
        // to 39, a run four half-lives long.
        let y = (!(20..40).contains(&i)).then(|| 1.0 + x[0] + 0.1 * lcg(&mut s));
        let w = if i % 10 == 9 { 1000.0 } else { 1.0 };
        opened += usize::from(m.keeps_history(m.moments_gap() + 1.0));
        past += usize::from(m.keeps_history(super::unobserved(&m.elapsed) + 1.0));
        m.step(&x, &[y], 1.0, w);
        plain.update(&[1.0, x[0], x[1]], lam, w);
        assert_eq!(m.stats, plain, "row {i}");
    }
    assert_eq!(opened, 0, "the floor opened on {opened} rows");
    assert_eq!(past, 16, "the run of missing targets");
}

/// KAL-6's gap in the state: written only where the last row that moved
/// the moments observed no target, so a state whose did keeps its bytes,
/// and after the covariance form where it is written, so the positional
/// encoding reads each in its place. A state without it, as every state of
/// schema 71 and earlier, reads the clock since a target was observed.
#[test]
fn the_moments_gap_is_written_where_it_is_not_the_clock_since_a_target() {
    for form in [CovarianceForm::Covariance, CovarianceForm::SquareRoot] {
        let mut cfg = plain_cfg(0.5, 5.0);
        cfg.standardize = true;
        let mut m = Kalman::new(cfg)
            .unwrap()
            .with_covariance_form(form)
            .unwrap();
        let mut s = 4u64;
        for i in 0..12 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = (i < 8).then(|| 1.0 + x[0]);
            m.step(&x, &[y], 1.5, 1.0);
            let named = rmp_serde::to_vec_named(&m).unwrap();
            let v = rmpv::decode::read_value(&mut named.as_slice()).unwrap();
            let gap = v
                .as_map()
                .unwrap()
                .iter()
                .find(|(k, _)| k.as_str() == Some("moments_gap"))
                .map(|(_, v)| v.as_f64().unwrap());
            if i < 8 {
                assert_eq!(gap, None, "{form:?}, row {i}");
            } else {
                assert_eq!(gap, Some(0.0), "{form:?}, row {i}");
                assert_eq!(super::unobserved(&m.elapsed), 1.5 * (i - 7) as f64);
            }
            for bytes in [named.clone(), rmp_serde::to_vec(&m).unwrap()] {
                let back: Kalman = rmp_serde::from_slice(&bytes).unwrap();
                assert_eq!(back.moments_gap(), m.moments_gap(), "{form:?}, row {i}");
                assert_eq!(back.covariance_form(), form);
                assert_eq!(rmp_serde::to_vec_named(&back).unwrap(), named);
            }
        }
        // Without the field: the clock since a target was observed.
        let named = rmp_serde::to_vec_named(&m).unwrap();
        let mut v = rmpv::decode::read_value(&mut named.as_slice()).unwrap();
        if let rmpv::Value::Map(fields) = &mut v {
            fields.retain(|(k, _)| k.as_str() != Some("moments_gap"));
        }
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &v).unwrap();
        let before: Kalman = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(before.moments_gap(), 6.0, "{form:?}");
        // A gap that is not a clock is damaged.
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let mut v = v.clone();
            if let rmpv::Value::Map(fields) = &mut v {
                fields.push(("moments_gap".into(), bad.into()));
            }
            let mut bytes = Vec::new();
            rmpv::encode::write_value(&mut bytes, &v).unwrap();
            assert!(rmp_serde::from_slice::<Kalman>(&bytes).is_err(), "{bad}");
        }
    }
}
