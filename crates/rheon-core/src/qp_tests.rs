use super::*;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn spd_matrix(s: &mut u64, d: usize, ridge: f64) -> Vec<f64> {
    let m = d + 3;
    let mut x = vec![0.0; m * d];
    for v in &mut x {
        *v = lcg(s);
    }
    let mut h = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = 0.0;
            for r in 0..m {
                sum += x[r * d + i] * x[r * d + j];
            }
            h[i * d + j] = sum / m as f64;
        }
        h[i * d + i] += ridge;
    }
    h
}

#[test]
fn simplex_box_and_nonneg_satisfy_exact_kkt_across_random_problems() {
    let mut s = 42u64;
    let mut work = QpWork::default();
    for trial in 0..200 {
        let d = 1 + trial % 12;
        let ridge = if trial % 5 == 0 { 0.0 } else { 1e-4 };
        let h = spd_matrix(&mut s, d, ridge);
        let c: Vec<f64> = (0..d).map(|_| 2.0 * lcg(&mut s)).collect();
        let (lo, hi, sum) = match trial % 4 {
            0 => (vec![0.0; d], vec![f64::INFINITY; d], Some(1.0)), // simplex
            1 => (vec![0.0; d], vec![f64::INFINITY; d], None),      // non-negative
            2 => (vec![-0.4; d], vec![0.4; d], None),               // box
            _ => {
                // box + sum
                let hi_val = (1.5 / d as f64).max(0.2);
                (vec![0.0; d], vec![hi_val; d], Some(1.0))
            }
        };
        let con = Constraint {
            lo: lo.clone(),
            hi: hi.clone(),
            sum,
        };
        con.validate(d, "test").unwrap();
        let mut x = vec![0.0; d];
        let out = solve_ridge_qp(
            &h, &c, d, 0.0, false, None, None, &con, None, &mut work, &mut x,
        );
        assert_eq!(out.status, QpStatus::Converged, "trial {trial} d={d}");
        let eq_a = vec![1.0; d];
        let eq = sum.map(|rhs| (eq_a.as_slice(), rhs));
        let kkt = kkt_residuals(&h, &c, d, &lo, &hi, eq, &x);
        assert!(
            kkt.max_err() < 1e-10,
            "trial {trial} d={d}: kkt={kkt:?} x={x:?}"
        );

        // Warm start from a random perturbed point matches the cold start.
        let warm: Vec<f64> = (0..d).map(|_| 3.0 * lcg(&mut s)).collect();
        let mut x_warm = vec![0.0; d];
        let out2 = solve_ridge_qp(
            &h,
            &c,
            d,
            0.0,
            false,
            None,
            None,
            &con,
            Some(&warm),
            &mut work,
            &mut x_warm,
        );
        assert_eq!(out2.status, QpStatus::Converged);
        if ridge > 0.0 {
            for i in 0..d {
                assert!(
                    (x[i] - x_warm[i]).abs() < 1e-12,
                    "trial {trial} i={i}: warm vs cold mismatch {} vs {}",
                    x[i],
                    x_warm[i]
                );
            }
        }
    }
}

#[test]
fn degenerate_vertex_box_plus_sum_escapes_and_converges() {
    // k = 3, lo = 0, hi = 0.5, sum = 1.0: starting from vertex (0.5, 0.5, 0.0)
    // where ALL variables sit at bounds!
    let h = vec![
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0,
    ];
    let con = Constraint {
        lo: vec![0.0, 0.0, 0.0],
        hi: vec![0.5, 0.5, 0.5],
        sum: Some(1.0),
    };
    let mut work = QpWork::default();

    // Case 1: escapes degenerate vertex (0.5, 0.5, 0.0) to interior face (0.15, 0.35, 0.5).
    let c1 = vec![0.0, 0.2, 0.8];
    let mut x1 = vec![0.0; 3];
    let out1 = solve_ridge_qp(
        &h,
        &c1,
        3,
        0.0,
        false,
        None,
        None,
        &con,
        Some(&[0.5, 0.5, 0.0]),
        &mut work,
        &mut x1,
    );
    assert_eq!(out1.status, QpStatus::Converged);
    assert!(
        (x1[0] - 0.15).abs() < 1e-12 && (x1[1] - 0.35).abs() < 1e-12 && (x1[2] - 0.5).abs() < 1e-12,
        "x1={x1:?}"
    );
    let kkt1 = kkt_residuals(&h, &c1, 3, &con.lo, &con.hi, Some((&[1.0; 3], 1.0)), &x1);
    assert!(kkt1.max_err() < 1e-12, "kkt1={kkt1:?}");

    // Case 2: escapes degenerate vertex (0.5, 0.5, 0.0) to another vertex (0.0, 0.5, 0.5).
    let c2 = vec![-1.0, 0.8, 0.9];
    let mut x2 = vec![0.0; 3];
    let out2 = solve_ridge_qp(
        &h,
        &c2,
        3,
        0.0,
        false,
        None,
        None,
        &con,
        Some(&[0.5, 0.5, 0.0]),
        &mut work,
        &mut x2,
    );
    assert_eq!(out2.status, QpStatus::Converged);
    assert!(
        (x2[0] - 0.0).abs() < 1e-12 && (x2[1] - 0.5).abs() < 1e-12 && (x2[2] - 0.5).abs() < 1e-12,
        "x2={x2:?}"
    );
    let kkt2 = kkt_residuals(&h, &c2, 3, &con.lo, &con.hi, Some((&[1.0; 3], 1.0)), &x2);
    assert!(kkt2.max_err() < 1e-12, "kkt2={kkt2:?}");
}

#[test]
fn collinear_and_constant_features_at_zero_ridge_solve_cleanly() {
    let mut work = QpWork::default();

    // Case 1: duplicated features x1 == x0, ridge = 0, simplex.
    // H = [[2, 2, 0.5], [2, 2, 0.5], [0.5, 0.5, 1]], c = [2, 2, 0.8].
    let h_dup = vec![
        2.0, 2.0, 0.5, //
        2.0, 2.0, 0.5, //
        0.5, 0.5, 1.0,
    ];
    let c_dup = vec![2.0, 2.0, 0.8];
    let con = Constraint {
        lo: vec![0.0; 3],
        hi: vec![f64::INFINITY; 3],
        sum: Some(1.0),
    };
    let mut x = vec![0.0; 3];
    let out = solve_ridge_qp(
        &h_dup, &c_dup, 3, 0.0, false, None, None, &con, None, &mut work, &mut x,
    );
    assert_eq!(out.status, QpStatus::Converged);
    let kkt = kkt_residuals(
        &h_dup,
        &c_dup,
        3,
        &con.lo,
        &con.hi,
        Some((&[1.0; 3], 1.0)),
        &x,
    );
    assert!(kkt.max_err() < 1e-11, "dup kkt={kkt:?} x={x:?}");

    // Case 2: constant feature (row/col 0 is zero) + active feature, simplex.
    // True optimal beta_1 = 0.3, beta_0 absorbs slack 0.7!
    let h_const = vec![
        0.0, 0.0, //
        0.0, 1.0,
    ];
    let c_const = vec![0.0, 0.3];
    let con2 = Constraint {
        lo: vec![0.0; 2],
        hi: vec![f64::INFINITY; 2],
        sum: Some(1.0),
    };
    let mut x2 = vec![0.0; 2];
    let out2 = solve_ridge_qp(
        &h_const, &c_const, 2, 0.0, false, None, None, &con2, None, &mut work, &mut x2,
    );
    assert_eq!(out2.status, QpStatus::Converged);
    assert!(
        (x2[0] - 0.7).abs() < 1e-12 && (x2[1] - 0.3).abs() < 1e-12,
        "x2={x2:?}"
    );
    let kkt2 = kkt_residuals(
        &h_const,
        &c_const,
        2,
        &con2.lo,
        &con2.hi,
        Some((&[1.0; 2], 1.0)),
        &x2,
    );
    assert!(kkt2.max_err() < 1e-12, "const kkt={kkt2:?}");

    // Case 3: standardized with constant feature + ridge > 0 + simplex.
    let mut x3 = vec![0.0; 2];
    let out3 = solve_ridge_qp(
        &h_const,
        &c_const,
        2,
        0.1,
        true,
        Some(&[1.0, 1.0]),
        None,
        &con2,
        None,
        &mut work,
        &mut x3,
    );
    assert_eq!(out3.status, QpStatus::Converged);
    // Active col has H_11 = 1 + 0.1 = 1.1, c_1 = 0.3 => beta_1 = 0.3 / 1.1, beta_0 = 1 - beta_1.
    let want1 = 0.3 / 1.1;
    assert!(
        (x3[1] - want1).abs() < 1e-12 && (x3[0] - (1.0 - want1)).abs() < 1e-12,
        "x3={x3:?}"
    );
}

#[test]
fn extreme_feature_scales_solve_to_full_precision() {
    // Feature 0 at scale 1e-5, feature 1 at scale 1e5, simplex constraint.
    let scales = [1e-5, 1e5];
    // Correlation matrix [[1, 0.3], [0.3, 1]] scaled by `scales`.
    let h = vec![
        scales[0] * scales[0],
        0.3 * scales[0] * scales[1],
        0.3 * scales[0] * scales[1],
        scales[1] * scales[1],
    ];
    let beta_true = [0.6, 0.4];
    let c = vec![
        h[0] * beta_true[0] + h[1] * beta_true[1],
        h[2] * beta_true[0] + h[3] * beta_true[1],
    ];
    let con = Constraint {
        lo: vec![0.0, 0.0],
        hi: vec![f64::INFINITY, f64::INFINITY],
        sum: Some(1.0),
    };
    let mut work = QpWork::default();
    let mut x = vec![0.0; 2];
    let out = solve_ridge_qp(
        &h, &c, 2, 0.0, false, None, None, &con, None, &mut work, &mut x,
    );
    assert_eq!(out.status, QpStatus::Converged);
    assert!(
        (x[0] - 0.6).abs() < 1e-11 && (x[1] - 0.4).abs() < 1e-11,
        "x={x:?}"
    );
}

#[test]
fn pinned_variables_and_unconstrained_intercept_slot_work() {
    // d = 3: slot 0 is unconstrained intercept (a[0] = 0), slot 1 is fixed at 0.25,
    // slot 2 has lo = 0, hi = inf, slope sum = 1.0 -> slot 2 must be 0.75.
    let h = vec![
        1.0, 0.2, -0.1, //
        0.2, 2.0, 0.3, //
        -0.1, 0.3, 1.5,
    ];
    let c = vec![1.0, 0.5, 0.8];
    let lo = vec![f64::NEG_INFINITY, 0.25, 0.0];
    let hi = vec![f64::INFINITY, 0.25, f64::INFINITY];
    let a = vec![0.0, 1.0, 1.0];
    let x_init = vec![0.0, 0.25, 0.75];
    let mut work = QpWork::default();
    let mut x = vec![0.0; 3];
    let out = solve_qp(
        &h,
        &c,
        3,
        &lo,
        &hi,
        Some((&a, 1.0)),
        &x_init,
        &mut work,
        &mut x,
    );
    assert_eq!(out.status, QpStatus::Converged);
    assert!((x[1] - 0.25).abs() < 1e-14);
    assert!((x[2] - 0.75).abs() < 1e-14);
    // Intercept satisfies H[0,0]*x[0] + H[0,1]*x[1] + H[0,2]*x[2] = c[0].
    let want0 = (c[0] - h[1] * 0.25 - h[2] * 0.75) / h[0];
    assert!((x[0] - want0).abs() < 1e-13, "x={x:?} want0={want0}");
}

#[test]
fn unbounded_zero_curvature_ray_and_nonfinite_inputs_fail_cleanly() {
    // H = 0, c = [1, 0], unconstrained -> unbounded below!
    let h = vec![0.0; 4];
    let c = vec![1.0, 0.0];
    let lo = vec![f64::NEG_INFINITY; 2];
    let hi = vec![f64::INFINITY; 2];
    let mut work = QpWork::default();
    let mut x = vec![0.0; 2];
    let out = solve_qp(&h, &c, 2, &lo, &hi, None, &[0.0, 0.0], &mut work, &mut x);
    assert_eq!(out.status, QpStatus::Failed);
    assert!(x.iter().all(|v| v.is_nan()));

    // Non-finite H -> fails cleanly.
    let h_nan = vec![f64::NAN, 0.0, 0.0, 1.0];
    let out2 = solve_qp(
        &h_nan,
        &c,
        2,
        &lo,
        &hi,
        None,
        &[0.0, 0.0],
        &mut work,
        &mut x,
    );
    assert_eq!(out2.status, QpStatus::Failed);
    assert!(x.iter().all(|v| v.is_nan()));
}

#[test]
fn warm_start_on_unchanged_active_set_finishes_in_one_iteration() {
    let mut s = 7u64;
    let h = spd_matrix(&mut s, 6, 1e-3);
    let c = vec![1.5, -0.8, 0.9, -0.3, 0.4, -1.2];
    let con = Constraint {
        lo: vec![0.0; 6],
        hi: vec![f64::INFINITY; 6],
        sum: Some(1.0),
    };
    let mut work = QpWork::default();
    let mut x1 = vec![0.0; 6];
    let out1 = solve_ridge_qp(
        &h, &c, 6, 0.0, false, None, None, &con, None, &mut work, &mut x1,
    );
    assert_eq!(out1.status, QpStatus::Converged);

    // Perturb `c` slightly without changing the active set -> 1 iteration!
    let c2: Vec<f64> = c.iter().map(|v| v + 1e-4).collect();
    let mut x2 = vec![0.0; 6];
    let out2 = solve_ridge_qp(
        &h,
        &c2,
        6,
        0.0,
        false,
        None,
        None,
        &con,
        Some(&x1),
        &mut work,
        &mut x2,
    );
    assert_eq!(out2.status, QpStatus::Converged);
    assert_eq!(
        out2.iters, 1,
        "expected 1 warm-started iter, got {}",
        out2.iters
    );
}
