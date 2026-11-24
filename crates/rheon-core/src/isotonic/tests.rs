use super::fit::{interp, isotonic, pava, pool_rows};
use super::*;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn cfg() -> IsotonicCfg {
    IsotonicCfg {
        n_targets: 1,
        increasing: true,
        y_min: None,
        y_max: None,
        out_of_bounds: OutOfBounds::Nan,
        min_weight: 0.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Halflife(f64::INFINITY),
        max_state_bytes: None,
    }
}

/// The isotonic fit by its definition, the min-max formula: at point `j`,
/// `max_{i <= j} min_{k >= j}` of the weighted mean of points `i..=k`
/// (Barlow et al. 1972, Theorem 1.4.4). Shares no arithmetic order with
/// PAV.
fn minmax(y: &[f64], w: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mean = |i: usize, k: usize| {
        let (s, ww) = (i..=k).fold((0.0, 0.0), |(s, ww), t| (s + w[t] * y[t], ww + w[t]));
        s / ww
    };
    (0..n)
        .map(|j| {
            (0..=j)
                .map(|i| (j..n).map(|k| mean(i, k)).fold(f64::INFINITY, f64::min))
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect()
}

fn fit_of(y: &[f64], w: &[f64]) -> Vec<f64> {
    let (mut y, mut w) = (y.to_vec(), w.to_vec());
    let mut r = vec![0; y.len() + 1];
    pava(&mut y, &mut w, &mut r);
    y
}

#[test]
fn pava_is_the_min_max_formula_and_scikit_learns_example() {
    // scikit-learn's docstring example of `isotonic_regression`.
    let y = [5.0, 3.0, 1.0, 2.0, 8.0, 10.0, 7.0, 9.0, 6.0, 4.0];
    let f = fit_of(&y, &[1.0; 10]);
    assert_eq!(&f[..4], &[2.75; 4]);
    for v in &f[4..] {
        assert!((v - 44.0 / 6.0).abs() < 1e-15, "{v}");
    }
    let mut s = 7u64;
    for n in 1..40 {
        for _ in 0..20 {
            let y: Vec<f64> = (0..n).map(|_| (lcg(&mut s) * 8.0).round() / 4.0).collect();
            let w: Vec<f64> = (0..n).map(|_| 0.1 + lcg(&mut s)).collect();
            let got = fit_of(&y, &w);
            let want = minmax(&y, &w);
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()), "{a} {b}");
            }
            assert!(got.windows(2).all(|p| p[0] <= p[1]), "non-decreasing");
        }
    }
}

#[test]
fn pava_meets_the_optimality_conditions() {
    // KKT: within each block of equal values every prefix mean is at least
    // the block's value and every suffix mean at most, and the weighted
    // residuals of each block sum to 0.
    let mut s = 11u64;
    for _ in 0..200 {
        let n = 1 + (lcg(&mut s) * 30.0) as usize;
        let y: Vec<f64> = (0..n).map(|_| lcg(&mut s) - 0.5).collect();
        let w: Vec<f64> = (0..n).map(|_| 0.5 + lcg(&mut s)).collect();
        let f = fit_of(&y, &w);
        let mut i = 0;
        while i < n {
            let mut k = i;
            while k + 1 < n && f[k + 1] == f[i] {
                k += 1;
            }
            let v = f[i];
            let mut acc = (0.0, 0.0);
            for t in i..=k {
                acc = (acc.0 + w[t] * y[t], acc.1 + w[t]);
                assert!(acc.0 / acc.1 >= v - 1e-12, "a prefix below its block");
            }
            let mut acc = (0.0, 0.0);
            for t in (i..=k).rev() {
                acc = (acc.0 + w[t] * y[t], acc.1 + w[t]);
                assert!(acc.0 / acc.1 <= v + 1e-12, "a suffix above its block");
            }
            i = k + 1;
        }
    }
}

#[test]
fn decreasing_is_the_increasing_fit_of_the_reversed_sequence() {
    let y = [1.0, 3.0, 2.0, 0.5, 0.7, -1.0];
    let w = [1.0, 2.0, 1.0, 1.0, 3.0, 1.0];
    let (mut a, mut aw, mut r) = (y.to_vec(), w.to_vec(), vec![]);
    isotonic(
        &mut a,
        &mut aw,
        &mut r,
        false,
        f64::NEG_INFINITY,
        f64::INFINITY,
    );
    let mut rev: Vec<f64> = y.iter().rev().copied().collect();
    let revw: Vec<f64> = w.iter().rev().copied().collect();
    rev = fit_of(&rev, &revw);
    rev.reverse();
    assert_eq!(a, rev);
    assert!(a.windows(2).all(|p| p[0] >= p[1]));
    // `[3, 3, 2, 0.6, 0.6, -1]`-like: the first pool is 1 and 3 together.
    assert_eq!(a[0], a[1]);
    // The clip after the fit, as `np.clip`.
    let (mut c, mut cw) = (y.to_vec(), w.to_vec());
    isotonic(&mut c, &mut cw, &mut r, false, 0.0, 2.0);
    for (v, u) in c.iter().zip(&a) {
        assert_eq!(*v, u.clamp(0.0, 2.0));
    }
}

#[test]
fn ties_pool_from_zero_in_the_rows_order() {
    let x = [0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
    let y = [-0.0, -0.0, 3.0, 1.0, 2.0, 4.0];
    let w = [1.0, 2.0, 0.0, 1.0, 1.0, 0.5];
    let (mut ux, mut uy, mut uw) = (vec![], vec![], vec![]);
    pool_rows(&x, &y, &w, &mut ux, &mut uy, &mut uw);
    // The weight-0 row at 1 is no row; -0 pooled from +0 is +0, as
    // scikit-learn's `current_y = 0` start makes it.
    assert_eq!(ux, vec![0.0, 2.0]);
    assert_eq!(uy[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(uw, vec![3.0, 2.5]);
    assert_eq!(uy[1], (0.0 + 1.0 + 2.0 + 4.0 * 0.5) / 2.5);
}

#[test]
fn the_read_is_numpys_interp_with_scikit_learns_bounds() {
    let fx = [0.0, 1.0, 3.0];
    let fy = [0.5, 0.5, 2.0];
    let nan = OutOfBounds::Nan;
    assert!(interp(&fx, &fy, -0.1, nan).is_nan());
    assert!(interp(&fx, &fy, 3.1, nan).is_nan());
    assert_eq!(interp(&fx, &fy, 0.0, nan), 0.5);
    assert_eq!(interp(&fx, &fy, 0.7, nan), 0.5);
    assert_eq!(interp(&fx, &fy, 2.0, nan), 0.75 * 1.0 + 0.5);
    assert_eq!(interp(&fx, &fy, 3.0, nan), 2.0);
    let clip = OutOfBounds::Clip;
    assert_eq!(interp(&fx, &fy, -5.0, clip), 0.5);
    assert_eq!(interp(&fx, &fy, 9.0, clip), 2.0);
    // One distinct `x` is its value everywhere, under either.
    assert_eq!(interp(&[1.0], &[4.0], -9.0, nan), 4.0);
    assert_eq!(interp(&[1.0], &[4.0], 9.0, clip), 4.0);
    assert!(interp(&[], &[], 0.0, clip).is_nan());
    // The last `x` exactly, where the line from the left could round
    // otherwise: numpy's special case.
    let fx = [0.0, 0.1, 0.3];
    let fy = [0.0, 0.1, 0.7];
    assert_eq!(interp(&fx, &fy, 0.3, nan), 0.7);
    // An infinite slope: at a point held, its value; between, numpy's
    // second try from the right end where the first is NaN.
    let fx = [0.0, 1e-308, 1.0];
    let fy = [0.0, 1.0, 1.0];
    assert_eq!(interp(&fx, &fy, 0.0, nan), 0.0);
    let fx = [-1e-300, 1e-300];
    let fy = [-1e300, 1e300];
    assert_eq!(interp(&fx, &fy, 0.0, nan), f64::INFINITY, "numpy's answer");
}

type Row = (f64, f64, f64);

/// The batch fit over `rows`, as scikit-learn makes it: sorted by `(x,
/// y)`, stable, pooled, fitted, read at `x`.
fn batch(rows: &[Row], c: &IsotonicCfg, x: f64) -> f64 {
    let mut rows: Vec<Row> = rows.iter().copied().filter(|r| r.2 > 0.0).collect();
    rows.sort_by(|a, b| (a.0, a.1).partial_cmp(&(b.0, b.1)).unwrap());
    let xs: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let ys: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let ws: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let (mut ux, mut uy, mut uw) = (vec![], vec![], vec![]);
    pool_rows(&xs, &ys, &ws, &mut ux, &mut uy, &mut uw);
    let (lo, hi) = c.clip();
    isotonic(&mut uy, &mut uw, &mut vec![], c.increasing, lo, hi);
    interp(&ux, &uy, x, c.out_of_bounds)
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// A stream on `levels` distinct `x` (continuous where 0), weights 1 or
/// varied, a target with a trend and noise.
fn stream(n: usize, levels: usize, weighted: bool, seed: u64) -> Vec<Row> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let u = lcg(&mut s);
            let x = if levels == 0 {
                u * 10.0
            } else {
                (u * levels as f64).floor() * 0.37
            };
            let y = 0.3 * x + lcg(&mut s) - 0.5;
            let w = if weighted { 0.25 + lcg(&mut s) } else { 1.0 };
            (x, y, w)
        })
        .collect()
}

#[test]
fn each_row_is_predicted_by_the_batch_fit_of_the_rows_before_it() {
    // Without ties every distinct `x` is one row, so the online sums are
    // the batch's: to the bit, either direction, either bound, the clip.
    for (increasing, oob, y_min) in [
        (true, OutOfBounds::Nan, None),
        (false, OutOfBounds::Clip, None),
        (true, OutOfBounds::Clip, Some(-0.2)),
    ] {
        let c = IsotonicCfg {
            increasing,
            out_of_bounds: oob,
            y_min,
            y_max: Some(2.5),
            ..cfg()
        };
        let rows = stream(300, 0, true, 3);
        let mut m = Isotonic::new(c.clone()).unwrap();
        for (i, r) in rows.iter().enumerate() {
            let p = m.step(&[r.0], &[Some(r.1)], 1.0, r.2).pred[0];
            let want = batch(&rows[..i], &c, r.0);
            assert!(same(p, want), "row {i}: {p} against {want}");
        }
    }
}

#[test]
fn ties_without_a_window_agree_within_their_summing_order() {
    // Ties are pooled as they arrive, where scikit-learn sums them in
    // ascending `y`: a few ulps.
    let c = cfg();
    let rows = stream(400, 12, true, 5);
    let mut m = Isotonic::new(c.clone()).unwrap();
    let mut worst = 0.0f64;
    for (i, r) in rows.iter().enumerate() {
        let p = m.step(&[r.0], &[Some(r.1)], 1.0, r.2).pred[0];
        let want = batch(&rows[..i], &c, r.0);
        if want.is_nan() {
            assert!(p.is_nan());
            continue;
        }
        worst = worst.max((p - want).abs() / want.abs().max(1e-300));
    }
    assert!(worst < 1e-14, "{worst}");
    assert_eq!(m.held(0), 12, "one point a distinct x");
}

#[test]
fn a_window_is_the_batch_fit_of_its_rows_to_the_bit() {
    // Each row held, ties summed in scikit-learn's order.
    for levels in [0, 7] {
        let c = IsotonicCfg {
            window_rows: Some(25),
            ..cfg()
        };
        let rows = stream(200, levels, true, 9);
        let mut m = Isotonic::new(c.clone()).unwrap();
        for (i, r) in rows.iter().enumerate() {
            let p = m.step(&[r.0], &[Some(r.1)], 1.0, r.2).pred[0];
            let want = batch(&rows[i.saturating_sub(25)..i], &c, r.0);
            assert!(
                same(p, want),
                "levels {levels}, row {i}: {p} against {want}"
            );
        }
        assert_eq!(m.window_len(), 25);
        assert!((m.n_eff() - rows[175..].iter().map(|r| r.2).sum::<f64>()).abs() < 1e-12);
    }
}

#[test]
fn a_clock_window_keeps_the_rows_of_its_span_under_either_edge() {
    for closed in [WindowClosed::Right, WindowClosed::Both] {
        let c = IsotonicCfg {
            window_clock: Some(10.0),
            closed,
            ..cfg()
        };
        let rows = stream(120, 5, false, 13);
        let mut m = Isotonic::new(c.clone()).unwrap();
        for (i, r) in rows.iter().enumerate() {
            // Row `i` at clock `i`: the rows within 10 of it, the one
            // exactly 10 back kept only under `Both`.
            let p = m.step(&[r.0], &[Some(r.1)], 1.0, r.2).pred[0];
            let keep = if closed == WindowClosed::Both { 10 } else { 9 };
            let want = batch(&rows[i.saturating_sub(keep)..i], &c, r.0);
            assert!(same(p, want), "{closed:?}, row {i}: {p} against {want}");
        }
    }
}

#[test]
fn a_decay_weighs_each_row_by_its_factor_across_many_rebases() {
    // The batch handed `0.5^((T - t_i) / h)` as scikit-learn is: one `exp2`
    // a row either way, the origin moved every 128 half-lives.
    let h = 3.0;
    let c = IsotonicCfg {
        decay: Decay::Halflife(h),
        ..cfg()
    };
    let rows = stream(3000, 9, false, 17);
    let mut m = Isotonic::new(c.clone()).unwrap();
    let mut worst = 0.0f64;
    let mut origins = 0;
    let mut origin = 0.0;
    for (i, r) in rows.iter().enumerate() {
        let p = m.step(&[r.0], &[Some(r.1)], 1.0, r.2).pred[0];
        if m.clock.origin != origin {
            origins += 1;
            origin = m.clock.origin;
        }
        // Only the last 1,000 half-lives' rows are weights a double holds.
        let lo = i.saturating_sub(2000);
        let past: Vec<Row> = (lo..i)
            .map(|k| (rows[k].0, rows[k].1, (-((i - 1 - k) as f64) / h).exp2()))
            .collect();
        let want = batch(&past, &c, r.0);
        if want.is_nan() {
            assert!(p.is_nan(), "row {i}");
            continue;
        }
        worst = worst.max((p - want).abs() / want.abs().max(1e-3));
    }
    assert!(origins >= 7, "{origins} origins");
    assert!(worst < 1e-12, "{worst}");
}

#[test]
fn a_decay_beside_a_window_weighs_the_windows_rows() {
    let h = 4.0;
    let c = IsotonicCfg {
        decay: Decay::Halflife(h),
        window_rows: Some(30),
        ..cfg()
    };
    let rows = stream(700, 6, false, 19);
    let mut m = Isotonic::new(c.clone()).unwrap();
    for (i, r) in rows.iter().enumerate() {
        let before = m.n_eff();
        let s = m.step(&[r.0], &[Some(r.1)], 1.0, r.2);
        assert_eq!(s.n_eff, before);
        let lo = i.saturating_sub(30);
        let past: Vec<Row> = (lo..i)
            .map(|k| (rows[k].0, rows[k].1, (-((i - 1 - k) as f64) / h).exp2()))
            .collect();
        let want = batch(&past, &c, r.0);
        let wsum: f64 = past.iter().map(|r| r.2).sum();
        assert!((s.n_eff - wsum).abs() < 1e-12 * (1.0 + wsum), "row {i}");
        if want.is_nan() {
            assert!(s.pred[0].is_nan());
        } else {
            assert!(
                (s.pred[0] - want).abs() < 1e-12 * (1.0 + want.abs()),
                "row {i}"
            );
        }
    }
}

#[test]
fn a_row_of_weight_zero_learns_nothing_first_row_too() {
    for c in [
        cfg(),
        IsotonicCfg {
            decay: Decay::Halflife(5.0),
            ..cfg()
        },
        IsotonicCfg {
            window_clock: Some(4.0),
            ..cfg()
        },
    ] {
        let mut m = Isotonic::new(c).unwrap();
        let s = m.step(&[1.0], &[Some(2.0)], 0.0, 0.0);
        assert!(s.pred[0].is_nan() && s.n_eff == 0.0);
        assert_eq!(m.held(0), 0);
        assert!(m.targets[0].fx.is_empty());
        m.step(&[1.0], &[Some(2.0)], 1.0, 1.0);
        let s = m.step(&[1.0], &[Some(9.0)], 1.0, 1e-320);
        assert_eq!(s.pred[0], 2.0);
        assert_eq!(m.held(0), 1);
        assert_eq!(m.predict(&[1.0], 1.0).pred[0], 2.0);
    }
}

#[test]
fn an_absent_target_learns_nothing_of_its_own() {
    let c = IsotonicCfg {
        n_targets: 2,
        ..cfg()
    };
    let mut m = Isotonic::new(c).unwrap();
    m.step(&[0.0], &[Some(1.0), None], 0.0, 1.0);
    m.step(&[1.0], &[Some(2.0), Some(5.0)], 1.0, 1.0);
    let s = m.step(&[0.5], &[None, None], 1.0, 1.0);
    assert_eq!(s.pred[0], 1.5);
    assert_eq!(s.pred[1], 5.0, "one point: its value everywhere");
    assert_eq!(m.target_weight(0), 2.0);
    assert_eq!(m.target_weight(1), 1.0);
    assert_eq!(m.n_eff(), 3.0);
}

#[test]
fn min_weight_withholds_on_each_side_of_the_floor() {
    let c = IsotonicCfg {
        min_weight: 2.0,
        out_of_bounds: OutOfBounds::Clip,
        ..cfg()
    };
    let mut m = Isotonic::new(c).unwrap();
    m.step(&[0.0], &[Some(1.0)], 0.0, 1.0);
    // Weight 1: below the floor.
    assert!(m.predict(&[0.0], 1.0).pred[0].is_nan());
    m.step(&[1.0], &[Some(3.0)], 1.0, 1.0);
    // Weight 2: at the floor.
    assert_eq!(m.predict(&[0.0], 1.0).pred[0], 1.0);
    // The floor lowered for a read, and put back.
    let mut m2 = Isotonic::new(IsotonicCfg {
        min_weight: 5.0,
        ..m.cfg().clone()
    })
    .unwrap();
    m2.step(&[0.0], &[Some(1.0)], 0.0, 1.0);
    let read = m2.predict_with_floor(&[0.0], &[None], 1.0, 0.0);
    assert_eq!(read.pred[0], 1.0);
    assert_eq!(m2.cfg().min_weight, 5.0);
}

#[test]
fn the_state_cap_warns_at_half_and_refuses_past_it() {
    // Without a window each new distinct `x` is 40 bytes.
    let fixed = cfg().fixed_bytes();
    let n = 200;
    let cap = fixed + n * ISOTONIC_POINT_BYTES;
    let c = IsotonicCfg {
        max_state_bytes: Some(cap),
        ..cfg()
    };
    let mut m = Isotonic::new(c).unwrap();
    let mut warned = 0;
    for i in 0..n {
        m.step(&[i as f64], &[Some(i as f64)], 1.0, 1.0);
        let warn = m.take_state_warning();
        warned += usize::from(warn.is_some());
        // Half of the cap: the first row past it is said, once.
        let past_half = |n: u64| 2 * (fixed + n * ISOTONIC_POINT_BYTES) > cap;
        let first = past_half(i + 1) && !past_half(i);
        assert_eq!(warn.is_some(), first, "row {i}");
        assert!(
            m.state_over_cap().is_none(),
            "row {i}: at the cap is under it"
        );
    }
    assert_eq!(warned, 1);
    assert_eq!(m.bytes(), cap);
    // A tie adds nothing and is learned at the cap.
    m.step(&[3.0], &[Some(4.0)], 1.0, 1.0);
    assert!(m.state_over_cap().is_none());
    // A new `x` would pass it: not learned, nor anything after.
    m.step(&[n as f64], &[Some(1.0)], 1.0, 1.0);
    assert_eq!(m.state_over_cap(), Some((cap + ISOTONIC_POINT_BYTES, cap)));
    let held = m.held(0);
    m.step(&[3.0], &[Some(4.0)], 1.0, 1.0);
    assert_eq!(m.held(0), held);
    assert_eq!(m.n_eff(), n as f64 + 1.0);
}

#[test]
fn a_row_window_is_checked_at_the_call() {
    let one = IsotonicCfg {
        window_rows: Some(1),
        ..cfg()
    };
    let row = one.held_row_bytes();
    let base = one.fixed_bytes() - row;
    let at = IsotonicCfg {
        window_rows: Some(100),
        max_state_bytes: Some(base + 100 * row),
        ..cfg()
    };
    assert!(at.validate().is_ok());
    let past = IsotonicCfg {
        window_rows: Some(101),
        ..at
    };
    let e = past.validate().unwrap_err();
    assert!(e.contains("max_state_bytes"), "{e}");
}

#[test]
fn save_and_load_continue_to_the_bit() {
    for c in [
        cfg(),
        IsotonicCfg {
            window_rows: Some(17),
            decay: Decay::Halflife(2.0),
            ..cfg()
        },
        IsotonicCfg {
            window_clock: Some(9.0),
            increasing: false,
            ..cfg()
        },
    ] {
        let rows = stream(400, 8, true, 23);
        let mut a = Isotonic::new(c).unwrap();
        for r in &rows[..200] {
            a.step(&[r.0], &[Some(r.1)], 1.0, r.2);
        }
        let bytes = rmp_serde::to_vec(&a.state()).unwrap();
        let mut b = Isotonic::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        assert_eq!(a, b);
        for r in &rows[200..] {
            let (p, q) = (
                a.step(&[r.0], &[Some(r.1)], 1.0, r.2),
                b.step(&[r.0], &[Some(r.1)], 1.0, r.2),
            );
            assert!(same(p.pred[0], q.pred[0]) && p.n_eff == q.n_eff);
        }
        assert_eq!(
            rmp_serde::to_vec(&a.state()).unwrap(),
            rmp_serde::to_vec(&b.state()).unwrap()
        );
    }
}

#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let c = IsotonicCfg {
        window_rows: Some(5),
        ..cfg()
    };
    let mut m = Isotonic::new(c).unwrap();
    for i in 0..4 {
        m.step(&[i as f64], &[Some(1.0)], 1.0, 1.0);
    }
    let breaks: [fn(&mut Isotonic); 6] = [
        |m| m.targets[0].x.pop().map(|_| ()).unwrap_or(()),
        |m| m.targets.push(Target::default()),
        |m| m.targets[0].x.swap(0, 3),
        |m| m.ring.as_mut().unwrap().y[0] = 7.0,
        |m| m.ring = None,
        |m| m.targets[0].w[1] = -1.0,
    ];
    for (i, b) in breaks.iter().enumerate() {
        let mut bad = m.clone();
        b(&mut bad);
        assert!(Isotonic::restore(&bad.state()).is_err(), "break {i} loaded");
    }
    let mut pooled = Isotonic::new(cfg()).unwrap();
    pooled.step(&[0.0], &[Some(1.0)], 1.0, 1.0);
    pooled.step(&[1.0], &[Some(1.0)], 1.0, 1.0);
    pooled.targets[0].x[1] = 0.0;
    assert!(Isotonic::restore(&pooled.state()).is_err(), "a repeated x");
}

#[test]
fn the_settings_are_checked() {
    let bad = [
        IsotonicCfg {
            n_targets: 0,
            ..cfg()
        },
        IsotonicCfg {
            y_min: Some(2.0),
            y_max: Some(1.0),
            ..cfg()
        },
        IsotonicCfg {
            y_max: Some(f64::INFINITY),
            ..cfg()
        },
        IsotonicCfg {
            window_rows: Some(3),
            window_clock: Some(3.0),
            ..cfg()
        },
        IsotonicCfg {
            closed: WindowClosed::Both,
            ..cfg()
        },
        IsotonicCfg {
            window_rows: Some(0),
            ..cfg()
        },
        IsotonicCfg {
            window_clock: Some(0.0),
            ..cfg()
        },
        IsotonicCfg {
            min_weight: -1.0,
            ..cfg()
        },
        IsotonicCfg {
            max_state_bytes: Some(0),
            ..cfg()
        },
    ];
    for (i, c) in bad.into_iter().enumerate() {
        assert!(Isotonic::new(c).is_err(), "case {i} accepted");
    }
    assert!(
        Isotonic::new(IsotonicCfg {
            y_min: Some(1.0),
            y_max: Some(1.0),
            ..cfg()
        })
        .is_ok()
    );
}

#[test]
fn beyond_sample_is_the_null_the_fit_reads() {
    let mut m = Isotonic::new(cfg()).unwrap();
    assert!(m.beyond_sample(0.0));
    m.step(&[1.0], &[Some(1.0)], 0.0, 1.0);
    assert!(!m.beyond_sample(5.0), "one point reads everywhere");
    m.step(&[2.0], &[Some(1.0)], 1.0, 1.0);
    assert!(m.beyond_sample(2.5) && !m.beyond_sample(1.5));
    assert!(m.predict(&[2.5], 1.0).pred[0].is_nan());
    let mut c = Isotonic::new(IsotonicCfg {
        out_of_bounds: OutOfBounds::Clip,
        ..cfg()
    })
    .unwrap();
    c.step(&[1.0], &[Some(1.0)], 0.0, 1.0);
    c.step(&[2.0], &[Some(3.0)], 1.0, 1.0);
    assert!(!c.beyond_sample(9.0));
    assert_eq!(c.predict(&[9.0], 1.0).pred[0], 3.0);
}
