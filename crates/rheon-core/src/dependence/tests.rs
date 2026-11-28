use super::*;
use crate::ns_of_seconds;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = ((*s >> 11) as f64) / ((1u64 << 53) as f64);
    2.0 * u - 1.0
}

fn plain_mi_and_mm(xs: &[f64], ys: &[f64], ws: &[f64], xe: &[f64], ye: &[f64]) -> (f64, f64) {
    let bx = xe.len() + 1;
    let by = ye.len() + 1;
    let mut c = vec![0.0; bx * by];
    let mut all_unit = true;
    for ((&x, &y), &w) in xs.iter().zip(ys).zip(ws) {
        if !(w > 0.0 && x.is_finite() && y.is_finite()) {
            continue;
        }
        if w != 1.0 {
            all_unit = false;
        }
        let a = xe.partition_point(|e| *e <= x);
        let b = ye.partition_point(|e| *e <= y);
        c[a * by + b] += w;
    }
    let n: f64 = c.iter().sum();
    if !n.is_finite() || n <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let mut na = vec![0.0; bx];
    let mut nb = vec![0.0; by];
    for a in 0..bx {
        for b in 0..by {
            let v = c[a * by + b];
            na[a] += v;
            nb[b] += v;
        }
    }
    let hx: f64 = na
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| -(v / n) * libm::log(v / n))
        .sum();
    let hy: f64 = nb
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| -(v / n) * libm::log(v / n))
        .sum();
    let hxy: f64 = c
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| -(v / n) * libm::log(v / n))
        .sum();
    let mi = (hx + hy - hxy).max(0.0);
    let b_xy = c.iter().filter(|&&v| v > 0.0).count() as i64;
    let b_x = na.iter().filter(|&&v| v > 0.0).count() as i64;
    let b_y = nb.iter().filter(|&&v| v > 0.0).count() as i64;
    let mm = if all_unit {
        mi - (b_xy - b_x - b_y + 1) as f64 / (2.0 * n)
    } else {
        f64::NAN
    };
    (mi, mm)
}

fn brute_kendall(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let (mut s, mut n0, mut n1, mut n2) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..n {
        for j in (i + 1)..n {
            let sx = if xs[i] > xs[j] {
                1
            } else if xs[i] < xs[j] {
                -1
            } else {
                0
            };
            let sy = if ys[i] > ys[j] {
                1
            } else if ys[i] < ys[j] {
                -1
            } else {
                0
            };
            s += sx * sy;
            n0 += 1;
            n1 += i64::from(sx == 0);
            n2 += i64::from(sy == 0);
        }
    }
    let den = ((n0 - n1) as f64 * (n0 - n2) as f64).sqrt();
    if den > 0.0 { s as f64 / den } else { f64::NAN }
}

fn brute_xi(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap());
    let mut r = Vec::with_capacity(n);
    let mut l = Vec::with_capacity(n);
    for &i in &ord {
        let yi = ys[i];
        r.push(ys.iter().filter(|&&v| v <= yi).count() as i64);
        l.push(ys.iter().filter(|&&v| v >= yi).count() as i64);
    }
    let num: i64 = r.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    let den: i64 = 2 * l.iter().map(|&li| li * (n as i64 - li)).sum::<i64>();
    if den > 0 {
        1.0 - (n as f64 * num as f64) / (den as f64)
    } else {
        f64::NAN
    }
}

#[test]
fn cfg_validation_refuses_every_invalid_combination() {
    let base = DependenceCfg::default();
    assert!(base.validate().is_ok());

    // Dims and measures
    assert!(
        DependenceCfg {
            n_features: 0,
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            n_targets: 0,
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            measures: vec![],
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            measures: vec![DependenceMeasure::Mi, DependenceMeasure::Mi],
            ..base.clone()
        }
        .validate()
        .is_err()
    );

    // Decay incompatibilities
    assert!(
        DependenceCfg {
            measures: vec![DependenceMeasure::MiMm],
            decay: Decay::Halflife(50.0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            measures: vec![DependenceMeasure::Spearman],
            window_rows: Some(20),
            decay: Decay::Halflife(50.0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            measures: vec![DependenceMeasure::Mi],
            window_rows: Some(20),
            decay: Decay::Lam(0.95),
            ..base.clone()
        }
        .validate()
        .is_err()
    );

    // Ranks without window
    for m in [
        DependenceMeasure::Spearman,
        DependenceMeasure::Kendall,
        DependenceMeasure::Xi,
    ] {
        assert!(
            DependenceCfg {
                measures: vec![m],
                ..base.clone()
            }
            .validate()
            .is_err()
        );
    }

    // Windows and closed
    assert!(
        DependenceCfg {
            window_rows: Some(0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            window_clock: Some(0.0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            window_clock: Some(f64::INFINITY),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            window_rows: Some(10),
            window_clock: Some(10.0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            closed: WindowClosed::Both,
            ..base.clone()
        }
        .validate()
        .is_err()
    );

    // Bins & edges
    assert!(
        DependenceCfg {
            bins: 1,
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            target_bins: 1,
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            bin_warm_rows: Some(1),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            bin_warm_span: Some(0.0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            bin_warm_rows: Some(10),
            bin_warm_span: Some(10.0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            bin_edges: Some(vec![vec![1.0, 0.5]]),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            bin_edges: Some(vec![vec![0.0]]),
            target_bin_edges: Some(vec![vec![0.0]]),
            bin_warm_rows: Some(10),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            measures: vec![DependenceMeasure::Kendall],
            window_rows: Some(20),
            bin_warm_rows: Some(10),
            ..base.clone()
        }
        .validate()
        .is_err()
    );

    // Rank cadence
    assert!(
        DependenceCfg {
            measures: vec![DependenceMeasure::Kendall],
            window_rows: Some(20),
            rank_every_rows: Some(5),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            measures: vec![DependenceMeasure::Spearman],
            window_rows: Some(20),
            rank_every_rows: Some(0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            measures: vec![DependenceMeasure::Spearman],
            window_rows: Some(20),
            rank_every_clock: Some(0.0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            measures: vec![DependenceMeasure::Spearman],
            window_rows: Some(20),
            rank_every_rows: Some(5),
            rank_every_clock: Some(5.0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );

    // Cap checks
    assert!(
        DependenceCfg {
            max_state_bytes: Some(0),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            max_state_bytes: Some(crate::clock_span::STATE_BYTES_CEILING + 1),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
    assert!(
        DependenceCfg {
            measures: vec![DependenceMeasure::Spearman],
            window_rows: Some(10_000),
            n_features: 10,
            n_targets: 1,
            max_state_bytes: Some(256 * 1024),
            ..base.clone()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn mi_and_miller_madow_match_definition_and_warmup_replays_to_the_bit() {
    let mut seed = 101u64;
    let n = 400;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = lcg(&mut seed);
        let y = x * x + 0.15 * lcg(&mut seed);
        xs.push(x);
        ys.push(y);
    }
    let warm = 80;
    let mut x_vals: Vec<(f64, f64)> = xs[..warm].iter().map(|&v| (v, 1.0)).collect();
    let mut y_vals: Vec<(f64, f64)> = ys[..warm].iter().map(|&v| (v, 1.0)).collect();
    let xe = crate::edges_from(BinRule::Quantile, 8, &mut x_vals);
    let ye = crate::edges_from(BinRule::Quantile, 8, &mut y_vals);

    let mut m_warm = Dependence::new(DependenceCfg {
        measures: vec![DependenceMeasure::Mi, DependenceMeasure::MiMm],
        bins: 8,
        target_bins: 8,
        bin_warm_rows: Some(warm),
        ..Default::default()
    })
    .unwrap();
    let mut m_fixed = Dependence::new(DependenceCfg {
        measures: vec![DependenceMeasure::Mi, DependenceMeasure::MiMm],
        bin_edges: Some(vec![xe.clone()]),
        target_bin_edges: Some(vec![ye.clone()]),
        ..Default::default()
    })
    .unwrap();

    let ws = vec![1.0; n];
    for i in 0..n {
        let sw = m_warm.step(&[xs[i]], &[Some(ys[i])], 1.0, 1.0);
        let sf = m_fixed.step(&[xs[i]], &[Some(ys[i])], 1.0, 1.0);
        if i < warm {
            assert!(sw.pred[0].is_nan() && sw.pred[1].is_nan());
        } else {
            // Replay matches pre-known edges to the bit!
            assert_eq!(sw.pred[0].to_bits(), sf.pred[0].to_bits(), "row {i} mi");
            assert_eq!(sw.pred[1].to_bits(), sf.pred[1].to_bits(), "row {i} mm");
            let (ref_mi, ref_mm) = plain_mi_and_mm(&xs[..i], &ys[..i], &ws[..i], &xe, &ye);
            assert!(
                (sw.pred[0] - ref_mi).abs() < 1e-13,
                "row {i}: {} vs {ref_mi}",
                sw.pred[0]
            );
            assert!(
                (sw.pred[1] - ref_mm).abs() < 1e-13,
                "row {i}: {} vs {ref_mm}",
                sw.pred[1]
            );
            // Miller-Madow is subtracted, so it must be strictly below plug-in when B_xy > B_x + B_y - 1.
            assert!(sw.pred[1] <= sw.pred[0] + 1e-15);
        }
    }
}

#[test]
fn decayed_mi_matches_longhand_across_scale_folds() {
    let mut seed = 202u64;
    let xe = vec![-0.5, 0.0, 0.5];
    let ye = vec![-0.3, 0.3];
    let half_life = 3.0;
    let lam = Decay::Halflife(half_life).factor(1.0);
    let mut m = Dependence::new(DependenceCfg {
        measures: vec![DependenceMeasure::Mi],
        decay: Decay::Halflife(half_life),
        bin_edges: Some(vec![xe.clone()]),
        target_bin_edges: Some(vec![ye.clone()]),
        ..Default::default()
    })
    .unwrap();

    let (bx, by) = (xe.len() + 1, ye.len() + 1);
    let mut longhand = vec![0.0; bx * by];
    let mut max_gap = 0.0_f64;

    for i in 0..1_200 {
        let x = lcg(&mut seed);
        let y = 0.7 * x + 0.3 * lcg(&mut seed);
        let w = 0.2 + 1.8 * (0.5 * (lcg(&mut seed) + 1.0));
        let step = m.step(&[x], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, w);
        if i > 0 {
            let (mut na, mut nb) = (Vec::new(), Vec::new());
            let (ref_mi, _) = mi::mi_from_slice(&longhand, bx, by, false, &mut na, &mut nb);
            max_gap = max_gap.max((step.pred[0] - ref_mi).abs());
            for v in &mut longhand {
                *v *= lam;
            }
        }
        let a = xe.partition_point(|e| *e <= x);
        let b = ye.partition_point(|e| *e <= y);
        longhand[a * by + b] += w;
    }
    assert!(max_gap < 1e-12, "max decayed MI gap {max_gap}");
}

#[test]
fn windowed_mi_recovers_from_1e100_weight_and_gates_mm_on_fractional_weights() {
    let xe = vec![-0.3, 0.3];
    let ye = vec![-0.3, 0.3];
    let mut m = Dependence::new(DependenceCfg {
        measures: vec![DependenceMeasure::Mi, DependenceMeasure::MiMm],
        bin_edges: Some(vec![xe.clone()]),
        target_bin_edges: Some(vec![ye.clone()]),
        window_rows: Some(10),
        ..Default::default()
    })
    .unwrap();
    let mut seed = 303u64;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ws = Vec::new();

    for i in 0usize..40 {
        let x = lcg(&mut seed);
        let y = x + 0.1 * lcg(&mut seed);
        let w = if i == 5 {
            1e100
        } else if i == 18 {
            0.5
        } else {
            1.0
        };
        let step = m.step(&[x], &[Some(y)], 1.0, w);
        if i > 0 {
            let start = i.saturating_sub(10);
            let (ref_mi, ref_mm) =
                plain_mi_and_mm(&xs[start..i], &ys[start..i], &ws[start..i], &xe, &ye);
            assert!(
                (step.pred[0] - ref_mi).abs() < 1e-12,
                "row {i}: mi {} vs {ref_mi}",
                step.pred[0]
            );
            if ref_mm.is_nan() {
                assert!(
                    step.pred[1].is_nan(),
                    "row {i}: expected NaN mi_mm under non-unit weight"
                );
            } else {
                assert!(
                    (step.pred[1] - ref_mm).abs() < 1e-12,
                    "row {i}: mm {} vs {ref_mm}",
                    step.pred[1]
                );
            }
        }
        xs.push(x);
        ys.push(y);
        ws.push(w);
    }
    // After row 18 leaves at row 28, unit weights are back so mi_mm must be finite again!
    let final_step = m.predict(&[0.0], 1.0);
    assert!(final_step.pred[1].is_finite());
}

#[test]
fn incremental_kendall_spearman_and_xi_match_brute_force_with_ties_and_missing_targets() {
    let mut seed = 404u64;
    let k = 3;
    let nt = 2;
    let w_rows = 25;
    let mut m = Dependence::new(DependenceCfg {
        n_features: k,
        n_targets: nt,
        measures: vec![
            DependenceMeasure::Kendall,
            DependenceMeasure::Spearman,
            DependenceMeasure::Xi,
        ],
        window_rows: Some(w_rows),
        ..Default::default()
    })
    .unwrap();

    let mut hist_x: Vec<Vec<f64>> = Vec::new();
    let mut hist_y: Vec<Vec<Option<f64>>> = Vec::new();

    for i in 0usize..120 {
        // Feature 0: continuous; feature 1: quantized (ties); feature 2: parabola of y0.
        let x0 = lcg(&mut seed);
        let x1 = (lcg(&mut seed) * 3.0).round();
        let x2 = lcg(&mut seed);
        let y0 = if i % 11 == 0 {
            None
        } else {
            Some((x2 * x2 + 0.1 * x0 + 0.05 * lcg(&mut seed)).clamp(-2.0, 2.0))
        };
        let y1 = if i % 13 == 0 {
            None
        } else {
            Some(((x1 + lcg(&mut seed)).round()).clamp(-3.0, 3.0))
        };
        let x = vec![x0, x1, x2];
        let y = vec![y0, y1];
        let step = m.step(&x, &y, 1.0, 1.0);

        let start = i.saturating_sub(w_rows);
        for t in 0..nt {
            for j in 0..k {
                let mut sx = Vec::new();
                let mut sy = Vec::new();
                for (hx, hy) in hist_x[start..i].iter().zip(&hist_y[start..i]) {
                    if let Some(yv) = hy[t] {
                        sx.push(hx[j]);
                        sy.push(yv);
                    }
                }
                let base_t = t * 3 * k;
                let got_kd = step.pred[base_t + j];
                let got_sp = step.pred[base_t + k + j];
                let got_xi = step.pred[base_t + 2 * k + j];
                let ref_kd = brute_kendall(&sx, &sy);
                let ref_xi = brute_xi(&sx, &sy);
                if ref_kd.is_nan() {
                    assert!(got_kd.is_nan(), "row {i} ({j},{t}) kd");
                } else {
                    assert!(
                        (got_kd - ref_kd).abs() < 1e-14,
                        "row {i} ({j},{t}): {got_kd} vs {ref_kd}"
                    );
                }
                if ref_xi.is_nan() {
                    assert!(got_xi.is_nan(), "row {i} ({j},{t}) xi");
                } else {
                    assert!(
                        (got_xi - ref_xi).abs() < 1e-14,
                        "row {i} ({j},{t}): {got_xi} vs {ref_xi}"
                    );
                }
                if sx.len() >= 2 && got_sp.is_finite() {
                    assert!((-1.0..=1.0).contains(&got_sp));
                }
            }
        }
        hist_x.push(x);
        hist_y.push(y);
    }

    // Nonlinear parabola x2 -> y0 should have small |spearman| and clearly positive xi!
    let pairs = m.read_pairs_now();
    let p_20 = pairs[2]; // t=0, j=2
    assert!(
        p_20.xi > 0.35,
        "parabola should yield strong Chatterjee xi, got {}",
        p_20.xi
    );
    assert!(
        p_20.spearman.abs() < 0.35,
        "parabola should have low Spearman, got {}",
        p_20.spearman
    );
}

#[test]
fn rank_every_rows_and_rank_every_clock_match_on_row_clock_and_read_pairs_now_is_fresh() {
    let mut m_rows = Dependence::new(DependenceCfg {
        measures: vec![DependenceMeasure::Spearman, DependenceMeasure::Xi],
        window_rows: Some(30),
        rank_every_rows: Some(7),
        ..Default::default()
    })
    .unwrap();
    let mut m_clock = Dependence::new(DependenceCfg {
        measures: vec![DependenceMeasure::Spearman, DependenceMeasure::Xi],
        window_rows: Some(30),
        rank_every_clock: Some(7.0),
        ..Default::default()
    })
    .unwrap();
    let mut m_every1 = Dependence::new(DependenceCfg {
        measures: vec![DependenceMeasure::Spearman, DependenceMeasure::Xi],
        window_rows: Some(30),
        rank_every_rows: Some(1),
        ..Default::default()
    })
    .unwrap();

    let mut seed = 505u64;
    for i in 0..50 {
        let x = lcg(&mut seed);
        let y = 2.0 * x + 0.1 * lcg(&mut seed);
        m_clock.stamp_next_on_rows(Stamp::Raw(i as f64, 0.0));
        let d = if i == 0 { 0.0 } else { 1.0 };
        let sr = m_rows.step(&[x], &[Some(y)], d, 1.0);
        let sc = m_clock.step(&[x], &[Some(y)], d, 1.0);
        let _s1 = m_every1.step(&[x], &[Some(y)], d, 1.0);
        assert_eq!(
            sr.pred[0].to_bits(),
            sc.pred[0].to_bits(),
            "row {i} spearman"
        );
        assert_eq!(sr.pred[1].to_bits(), sc.pred[1].to_bits(), "row {i} xi");
    }
    // After 50 rows (50 % 7 == 1, not a refresh row), `read_pairs_now` on m_rows
    // must match `m_every1.predict` to the bit!
    let now_pair = m_rows.read_pairs_now()[0];
    let exact = m_every1.predict(&[0.0], 1.0);
    assert_eq!(now_pair.spearman.to_bits(), exact.pred[0].to_bits());
    assert_eq!(now_pair.xi.to_bits(), exact.pred[1].to_bits());
}

#[test]
fn clock_window_closed_edges_predict_at_and_bin_warm_span() {
    let sec = ns_of_seconds(1.0).unwrap();
    let cfg_of = |closed| DependenceCfg {
        measures: vec![
            DependenceMeasure::Mi,
            DependenceMeasure::Kendall,
            DependenceMeasure::Spearman,
            DependenceMeasure::Xi,
        ],
        bins: 4,
        target_bins: 4,
        bin_warm_span: Some(4.0),
        window_clock: Some(5.0),
        closed,
        ..Default::default()
    };
    let mut m_right = Dependence::new(cfg_of(WindowClosed::Right)).unwrap();
    let mut m_both = Dependence::new(cfg_of(WindowClosed::Both)).unwrap();

    // Feed 6 rows at t = 0, 1, 2, 3, 4, 5 seconds.
    let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    for i in 0..6 {
        let st = Stamp::Ns(i as i128 * sec);
        let d = if i == 0 { 0.0 } else { 1.0 };
        m_right.stamp_next(st);
        m_both.stamp_next(st);
        let sr = m_right.step(&[xs[i]], &[Some(ys[i])], d, 1.0);
        let sb = m_both.step(&[xs[i]], &[Some(ys[i])], d, 1.0);
        // At t = 5s (i = 5), row at t = 0s is exactly 5s old:
        // Right drops it (4 rows in window), Both keeps it (5 rows in window)!
        if i == 5 {
            assert_eq!(sr.n_eff, 4.0);
            assert_eq!(sb.n_eff, 5.0);
            // Warm-up closed at t = 4s (i = 4), so MI is active by t = 5s!
            assert!(sr.pred[0].is_finite() && sb.pred[0].is_finite());
        }
    }
    // `predict_at` at t = 100s (far past window_clock = 5s) must see an empty window!
    let far = m_right.predict_at(&[0.0], 95.0, Stamp::Ns(100 * sec));
    assert_eq!(far.n_eff, 0.0);
    assert!(far.pred.iter().all(|v| v.is_nan()));
    // And `predict_at` must not have mutated state!
    assert_eq!(m_right.n_eff(), 5.0);
}

#[test]
fn held_cap_warns_once_at_half_and_refuses_atomically() {
    let sec = ns_of_seconds(1.0).unwrap();
    let cap = 2_048u64;
    let mut m = Dependence::new(DependenceCfg {
        measures: vec![DependenceMeasure::Kendall],
        window_clock: Some(100.0),
        max_state_bytes: Some(cap),
        ..Default::default()
    })
    .unwrap();

    let mut warnings = Vec::new();
    let mut hit_cap_at = None;
    let mut state_before_refusal = None;
    for i in 0..500 {
        let snap = m.state();
        m.stamp_next(Stamp::Ns(i as i128 * sec / 100));
        let _ = m.step(&[i as f64], &[Some(i as f64)], 0.01, 1.0);
        if let Some(w) = m.take_state_warning() {
            warnings.push(w);
        }
        if m.state_over_cap().is_some() {
            hit_cap_at = Some(i);
            state_before_refusal = Some(snap);
            break;
        }
    }
    assert_eq!(warnings.len(), 1, "expected exactly one half-cap warning");
    assert!(warnings[0].0 * 2 > cap && warnings[0].1 == cap);
    let hit = hit_cap_at.expect("expected cap to trigger");
    assert!(hit > 0);
    assert!(m.state_bytes() <= cap);

    // Atomicity: every state field EXCEPT `cap_over` must match `state_before_refusal`!
    let before = Dependence::restore(&state_before_refusal.unwrap()).unwrap();
    assert_eq!(m.rows_learned, before.rows_learned);
    assert_eq!(m.w_sum.to_bits(), before.w_sum.to_bits());
    assert_eq!(m.val_ring, before.val_ring);
    assert_eq!(m.kendall, before.kendall);
    assert_eq!(m.place, before.place);

    // Subsequent rows must also learn nothing.
    let over = m.state_over_cap();
    m.stamp_next(Stamp::Ns(999 * sec));
    let _ = m.step(&[1.0], &[Some(1.0)], 1.0, 1.0);
    assert_eq!(m.rows_learned, before.rows_learned);
    assert_eq!(m.state_over_cap(), over);
}

#[test]
fn state_roundtrip_at_every_stage_and_wrong_shape_is_refused() {
    let mut seed = 606u64;
    let mut m = Dependence::new(DependenceCfg {
        n_features: 2,
        n_targets: 2,
        measures: vec![
            DependenceMeasure::Mi,
            DependenceMeasure::MiMm,
            DependenceMeasure::Spearman,
            DependenceMeasure::Kendall,
            DependenceMeasure::Xi,
        ],
        bins: 6,
        target_bins: 6,
        bin_warm_rows: Some(15),
        window_rows: Some(20),
        ..Default::default()
    })
    .unwrap();

    // Test save/restore at empty (0), mid-warmup (8), post-warmup partial (17), full (30).
    for i in 0..35 {
        if matches!(i, 0 | 8 | 17 | 30) {
            let st = m.state();
            let bytes = rmp_serde::to_vec_named(&st).unwrap();
            let back_st: State = rmp_serde::from_slice(&bytes).unwrap();
            let back = Dependence::restore(&back_st).unwrap();
            assert_eq!(back, m);
            let x = [lcg(&mut seed), lcg(&mut seed)];
            let a = back.predict(&x, 1.0);
            let b = m.predict(&x, 1.0);
            assert!(a.n_eff.same_bits(&b.n_eff));
            assert!(a.pred.same_bits(&b.pred));
        }
        let x = [lcg(&mut seed), lcg(&mut seed)];
        let y = [Some(lcg(&mut seed)), Some(lcg(&mut seed))];
        m.step(&x, &y, 1.0, 1.0);
    }

    // STATE-3 corruption sweeps
    let corrupt_and_check = |mutate: &dyn Fn(&mut Dependence)| {
        let mut bad = m.clone();
        mutate(&mut bad);
        assert!(
            Dependence::restore(&bad.state()).is_err(),
            "corrupt state should have been refused"
        );
    };
    corrupt_and_check(&|d| d.w_sum = -1.0);
    corrupt_and_check(&|d| d.w_sum = f64::NAN);
    corrupt_and_check(&|d| {
        d.target_w.pop();
    });
    corrupt_and_check(&|d| d.target_rows.push(0));
    corrupt_and_check(&|d| d.mi_tables.as_mut().unwrap().scale = 2.0);
    corrupt_and_check(&|d| {
        d.mi_tables.as_mut().unwrap().weights.pop();
    });
    corrupt_and_check(&|d| d.mi_tables.as_mut().unwrap().weights[0] = -1.0);
    corrupt_and_check(&|d| {
        let mt = d.mi_tables.as_mut().unwrap();
        mt.counts[0] = 0;
        mt.weights[0] = 1.0;
    });
    corrupt_and_check(&|d| d.mi_tables.as_mut().unwrap().x_edges[0] = vec![2.0, 1.0]);
    corrupt_and_check(&|d| d.val_ring.as_mut().unwrap().ws.push_back(1.0));
    corrupt_and_check(&|d| d.val_ring.as_mut().unwrap().ws[0] = 0.0);
    corrupt_and_check(&|d| {
        d.kendall.as_mut().unwrap().n1[0] = d.kendall.as_ref().unwrap().n0[0] + 1
    });
    corrupt_and_check(&|d| {
        d.kendall.as_mut().unwrap().s[0] = d.kendall.as_ref().unwrap().n0[0] + 1
    });
    corrupt_and_check(&|d| {
        d.held_spearman.0.pop();
    });
    corrupt_and_check(&|d| d.held_xi.0.push(0.0));
    corrupt_and_check(&|d| d.cfg.max_state_bytes = Some(100));

    // Wrong model variant
    let other = crate::Audit::new(crate::AuditCfg {
        n_columns: 1,
        distinct_cap: 10,
        pairs: false,
        gap_cap: Some(100.0),
        has_clock: false,
        distinct: None,
    })
    .unwrap()
    .state();
    assert!(matches!(
        Dependence::restore(&other),
        Err(StateError::WrongModel {
            expected: "dependence",
            found: "audit"
        })
    ));
}

#[test]
fn reference_state_sizes_fit_under_256kb_default_cap() {
    let mi_decay = Dependence::new(DependenceCfg {
        n_features: 10,
        n_targets: 1,
        measures: vec![DependenceMeasure::Mi],
        decay: Decay::Halflife(50.0),
        bins: 16,
        target_bins: 16,
        bin_edges: Some(vec![(1..16).map(|i| i as f64).collect(); 10]),
        target_bin_edges: Some(vec![(1..16).map(|i| i as f64).collect()]),
        ..Default::default()
    })
    .unwrap();
    let mi_win = Dependence::new(DependenceCfg {
        n_features: 10,
        n_targets: 1,
        measures: vec![DependenceMeasure::Mi, DependenceMeasure::MiMm],
        bins: 16,
        target_bins: 16,
        bin_edges: Some(vec![(1..16).map(|i| i as f64).collect(); 10]),
        target_bin_edges: Some(vec![(1..16).map(|i| i as f64).collect()]),
        window_rows: Some(1_000),
        ..Default::default()
    })
    .unwrap();
    let ranks = Dependence::new(DependenceCfg {
        n_features: 10,
        n_targets: 1,
        measures: vec![
            DependenceMeasure::Spearman,
            DependenceMeasure::Kendall,
            DependenceMeasure::Xi,
        ],
        window_rows: Some(1_000),
        ..Default::default()
    })
    .unwrap();
    let combined = Dependence::new(DependenceCfg {
        n_features: 10,
        n_targets: 1,
        measures: vec![
            DependenceMeasure::Mi,
            DependenceMeasure::MiMm,
            DependenceMeasure::Spearman,
            DependenceMeasure::Kendall,
            DependenceMeasure::Xi,
        ],
        bins: 16,
        target_bins: 16,
        bin_warm_rows: Some(1_000),
        window_rows: Some(1_000),
        ..Default::default()
    })
    .unwrap();

    for (name, cfg) in [
        ("mi_decay", mi_decay.cfg()),
        ("mi_win", mi_win.cfg()),
        ("ranks", ranks.cfg()),
        ("combined", combined.cfg()),
    ] {
        let fb = cfg.fixed_bytes();
        assert!(fb < DEFAULT_STATE_BYTES, "{name}: {fb} bytes >= 256 KB");
    }
    assert!(mi_decay.cfg().fixed_bytes() < 25_000);
    assert!(mi_win.cfg().fixed_bytes() < 65_000);
    assert!(ranks.cfg().fixed_bytes() < 100_000);
    assert!(combined.cfg().fixed_bytes() < 135_000);
}

#[test]
fn default_bin_warm_rows_scales_with_bins_and_respects_state_cap() {
    // 1D edge precision ceil(62.5 * max(Bx, By)) vs 2D cell count 5 * Bx * By.
    assert_eq!(default_bin_warm_rows(4, 4), 250);
    assert_eq!(default_bin_warm_rows(8, 8), 500);
    assert_eq!(default_bin_warm_rows(16, 16), 1_280);
    assert_eq!(default_bin_warm_rows(4, 16), 1_000);
    assert_eq!(default_bin_warm_rows(32, 32), 5_120);

    let cfg_default = DependenceCfg {
        n_features: 1,
        n_targets: 1,
        bins: 16,
        target_bins: 16,
        ..Default::default()
    };
    assert_eq!(cfg_default.resolved_bin_warm_rows(), Some(1_280));

    let cfg_small = DependenceCfg {
        n_features: 1,
        n_targets: 1,
        bins: 4,
        target_bins: 4,
        ..Default::default()
    };
    assert_eq!(cfg_small.resolved_bin_warm_rows(), Some(250));

    // Explicit override stays untouched.
    let cfg_explicit = DependenceCfg {
        bins: 4,
        target_bins: 4,
        bin_warm_rows: Some(77),
        ..Default::default()
    };
    assert_eq!(cfg_explicit.resolved_bin_warm_rows(), Some(77));

    // Clamping by max_state_bytes prevents spurious warmup-hold refusal.
    let cfg_capped = DependenceCfg {
        n_features: 10,
        n_targets: 2,
        bins: 32,
        target_bins: 32,
        max_state_bytes: Some(256 * 1024),
        ..Default::default()
    };
    let w = cfg_capped.resolved_bin_warm_rows().unwrap();
    assert!(
        (2..5_120).contains(&w),
        "expected clamped warm rows, got {w}"
    );
    assert!(cfg_capped.validate().is_ok());
}
