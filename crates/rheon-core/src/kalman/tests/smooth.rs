//! The fixed-lag smoother (`kalman/smooth.rs`) against the smoother written
//! here from the pairs the filter held: each observation's prior and
//! filtered pair, carried into the caller's units, where the filter's
//! re-maps are no change of the coefficients, and the Rauch-Tung-Striebel
//! recursion run on them with gains from faer's partial-pivot LU
//! (`crate::oracle`), which shares no arithmetic with the ring's Cholesky.
//! Then what it must hold beside the filter: every prediction the filter's
//! to the bit, a row of weight 0 no observation, a state saved mid-ring
//! going on to the bit, the chain cut where the filter does not continue
//! itself, the filter's gate, and the refusals.

use super::*;
use crate::Stamp;
use crate::kalman::smooth::Logged;

#[derive(Clone)]
struct Row {
    x: Vec<f64>,
    y: Vec<Option<f64>>,
    d: f64,
    w: f64,
}

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn n(&mut self) -> f64 {
        let (a, b) = (self.u(), self.u());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

/// `n` rows of `kf` features and `m` targets on a drifting truth: clock
/// steps of 1, 2 and 3 (whole numbers, so a split gap sums exactly),
/// weights from 0.5 to 1.5, about one row in twelve at weight 0 (a target of
/// 1e300 on it) and one in twelve with each target null. `spread` scales
/// the features, growing along the stream when `grow` is set, so a
/// standardized filter re-maps.
fn stream(n: usize, kf: usize, m: usize, seed: u64, grow: bool) -> Vec<Row> {
    let mut r = Rng(seed);
    let mut beta: Vec<f64> = (0..=kf).map(|i| 1.0 - 0.3 * i as f64).collect();
    (0..n)
        .map(|i| {
            for b in &mut beta {
                *b += 0.02 * r.n();
            }
            let scale = if grow { 1.0 + 0.02 * i as f64 } else { 1.0 };
            let x: Vec<f64> = (0..kf).map(|f| scale * (r.n() + 0.3 * f as f64)).collect();
            let fit = beta[0]
                + x.iter()
                    .zip(&beta[1..])
                    .map(|(a, b)| a * b / scale)
                    .sum::<f64>();
            let u = r.u();
            let w = if u < 0.08 { 0.0 } else { 0.5 + r.u() };
            let y = (0..m)
                .map(|j| {
                    if r.u() < 0.08 {
                        None
                    } else if w == 0.0 {
                        Some(1e300)
                    } else {
                        Some((j as f64 + 1.0) * fit + 0.5 * r.n())
                    }
                })
                .collect();
            let d = if i == 0 {
                0.0
            } else {
                [1.0, 1.0, 2.0, 1.0, 3.0][(r.u() * 5.0) as usize]
            };
            Row { x, y, d, w }
        })
        .collect()
}

fn base(kf: usize, m: usize) -> KalmanCfg {
    KalmanCfg {
        min_weight_holds_learning: false,
        noise_units: true,
        n_features: kf,
        n_targets: m,
        fit_intercept: true,
        decay: Decay::Halflife(100.0),
        half_life: vec![30.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    }
}

/// What a run emitted: per row, its emission where it emitted any, and
/// each target's noise ratio after the row (what the variances read).
struct Run {
    model: Kalman,
    emitted: Vec<(usize, Smoothed, Vec<f64>)>,
    preds: Vec<Vec<u64>>,
}

fn run(model: Kalman, rows: &[Row], stamps: &[f64]) -> Run {
    let mut model = model;
    let mut emitted = vec![];
    let mut preds = vec![];
    for (r, row) in rows.iter().enumerate() {
        model.stamp_next(Stamp::from(stamps[r]));
        let out = model.step(&row.x, &row.y, row.d, row.w);
        preds.push(out.pred.iter().map(|p| p.to_bits()).collect());
        if let Some(s) = model.smoothed()
            && s.emitted.iter().any(|e| *e)
        {
            let ratios = (0..model.cfg.n_targets)
                .map(|j| model.own_noise_ratio(j))
                .collect();
            emitted.push((r, s.clone(), ratios));
        }
    }
    Run {
        model,
        emitted,
        preds,
    }
}

fn ids(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}

/// The map to the caller's units from an anchor `[hi, lo, scales]`, as
/// `coefficients` reads `b` out: `β_i = b_i / s_i`, `β_0 = b_0 − Σ β_i m_i`.
fn caller_map(anchor: &[f64], k: usize) -> Vec<f64> {
    let mut t = vec![0.0; k * k];
    for i in 0..k {
        t[i * k + i] = 1.0;
    }
    if anchor.is_empty() {
        return t;
    }
    for i in 1..k {
        let s = anchor[2 * k + i];
        t[i * k + i] = 1.0 / s;
        t[i] = -(anchor[i] + anchor[k + i]) / s;
    }
    t
}

fn mul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = (0..k).map(|l| a[i * k + l] * b[l * k + j]).sum();
        }
    }
    out
}

fn transpose(a: &[f64], k: usize) -> Vec<f64> {
    (0..k * k).map(|ij| a[(ij % k) * k + ij / k]).collect()
}

fn mat_vec(a: &[f64], v: &[f64]) -> Vec<f64> {
    let k = v.len();
    (0..k)
        .map(|i| (0..k).map(|l| a[i * k + l] * v[l]).sum())
        .collect()
}

/// A logged pair in the caller's units: each target's `β = T b` and `T P
/// Tᵀ`, and which slots `P` had sized.
struct Pair {
    b: Vec<Vec<f64>>,
    p: Vec<f64>,
    sized: Vec<bool>,
    stamp: Stamp,
}

fn in_caller_units(l: &Logged, k: usize) -> Pair {
    let t = caller_map(&l.anchor, k);
    Pair {
        b: l.b.chunks_exact(k).map(|b| mat_vec(&t, b)).collect(),
        p: mul(&mul(&t, &l.p, k), &transpose(&t, k), k),
        sized: (0..k).map(|i| l.p[i * k + i] != 0.0).collect(),
        stamp: l.stamp,
    }
}

/// The smoother written from the logged pairs of one covariance: for its
/// observation `t`, the pass back over `lag` observations, in the caller's
/// units, with `F` the map between observations there (`f(n)` from `n` to
/// `n + 1`, the identity but for a reverting slot unstandardized), the gain
/// `G = P_n Fᵀ (P⁻)⁻¹` on `P⁻`'s sized slots by faer's LU.
fn written(
    priors: &[Pair],
    posts: &[Pair],
    t: usize,
    lag: usize,
    f: &dyn Fn(usize) -> Vec<f64>,
    k: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut m = posts[t].b.clone();
    let mut v = posts[t].p.clone();
    for n in (t - lag..t).rev() {
        let (pn, pm) = (&posts[n], &priors[n + 1]);
        let a = mul(&pn.p, &transpose(&f(n), k), k);
        let s: Vec<usize> = (0..k)
            .filter(|&i| pm.p[i * k + i] != 0.0 && pm.sized[i])
            .collect();
        let ns = s.len();
        let pms: Vec<f64> = (0..ns * ns)
            .map(|ij| pm.p[s[ij / ns] * k + s[ij % ns]])
            .collect();
        let inv = crate::oracle::inverse(&pms);
        let mut g = vec![0.0; k * k];
        for r in 0..k {
            for (c, &sc) in s.iter().enumerate() {
                g[r * k + sc] = (0..ns).map(|l| a[r * k + s[l]] * inv[l * ns + c]).sum();
            }
        }
        for (mt, (bn, bm)) in m.iter_mut().zip(pn.b.iter().zip(&pm.b)) {
            let diff: Vec<f64> = mt.iter().zip(bm).map(|(a, b)| a - b).collect();
            let gd = mat_vec(&g, &diff);
            *mt = bn.iter().zip(gd).map(|(a, b)| a + b).collect();
        }
        let dv: Vec<f64> = v.iter().zip(&pm.p).map(|(a, b)| a - b).collect();
        let gvg = mul(&mul(&g, &dv, k), &transpose(&g, k), k);
        v = pn.p.iter().zip(gvg).map(|(a, b)| a + b).collect();
    }
    (m, v)
}

/// The largest difference between `got` and `want` over `1 + |want|`,
/// NaN where `want`'s slot is unsized and nowhere else.
fn worst(got: &[f64], want: &[f64], sized: &[bool]) -> f64 {
    let mut e: f64 = 0.0;
    for ((g, w), s) in got.iter().zip(want).zip(sized) {
        if !s {
            assert!(g.is_nan(), "an unsized slot emitted {g}");
            continue;
        }
        assert!(g.is_finite(), "a sized slot emitted {g}");
        e = e.max((g - w).abs() / (1.0 + w.abs()));
    }
    e
}

/// Every emission of `r` against the written smoother, and the tail read
/// from the state at the end against the same at the stream's last
/// observation; returns the worst relative difference and the count of
/// values held.
fn check(r: &Run, lag: usize, f: &dyn Fn(usize, usize) -> Vec<f64>) -> (f64, usize) {
    let k = r.model.cfg.k_total();
    let n_p = r.model.p.len();
    let log = r.model.smooth_log();
    let mut err: f64 = 0.0;
    let mut held = 0;
    for pi in 0..n_p {
        let priors: Vec<Pair> = log
            .iter()
            .filter(|l| l.pi == pi && l.prior)
            .map(|l| in_caller_units(l, k))
            .collect();
        let posts: Vec<Pair> = log
            .iter()
            .filter(|l| l.pi == pi && !l.prior)
            .map(|l| in_caller_units(l, k))
            .collect();
        assert_eq!(priors.len(), posts.len());
        let targets: Vec<usize> = r.model.smooth_targets(pi).collect();
        let fp = |n: usize| f(pi, n);
        for t in lag..posts.len() {
            let (want_m, want_v) = written(&priors, &posts, t, lag, &fp, k);
            let s = t - lag;
            // The row that emitted it is observation `t`'s.
            let (_, out, ratios) = r
                .emitted
                .iter()
                .find(|(_, o, _)| targets.iter().any(|&j| o.stamp[j] == Some(posts[s].stamp)))
                .expect("every observation a lag back is emitted");
            for (ti, &j) in targets.iter().enumerate() {
                assert!(out.emitted[j]);
                if out.held[j] != SMOOTH_STOOD {
                    held += 1;
                    continue;
                }
                let got = &out.coef[j * k..(j + 1) * k];
                err = err.max(worst(got, &want_m[ti], &posts[s].sized));
                if r.model.smooth_se() {
                    let want: Vec<f64> = (0..k).map(|i| want_v[i * k + i] * ratios[j]).collect();
                    err = err.max(worst(&out.var[j * k..(j + 1) * k], &want, &posts[s].sized));
                }
            }
        }
        // The tail: the last observations at the stream's end.
        let tail = r.model.smoothed_tail().unwrap();
        let last = posts.len() - 1;
        for back in 0..lag.min(posts.len()) {
            let s = last - back;
            let (want_m, want_v) = written(&priors, &posts, last, back, &fp, k);
            for (ti, &j) in targets.iter().enumerate() {
                let row = tail
                    .iter()
                    .find(|row| row.target == j && row.stamp == posts[s].stamp)
                    .expect("each of the last observations in the tail");
                if row.held != SMOOTH_STOOD {
                    continue;
                }
                err = err.max(worst(&row.coef, &want_m[ti], &posts[s].sized));
                if r.model.smooth_se() {
                    let want: Vec<f64> = (0..k)
                        .map(|i| want_v[i * k + i] * r.model.own_noise_ratio(j))
                        .collect();
                    err = err.max(worst(&row.var, &want, &posts[s].sized));
                }
            }
        }
    }
    (err, held)
}

fn identity(k: usize) -> Vec<f64> {
    (0..k * k)
        .map(|ij| f64::from(u8::from(ij / k == ij % k)))
        .collect()
}

/// Unstandardized, the random walk: the ring is the smoother written from
/// the pairs, at every lag, over clock gaps, rows of weight 0 and null
/// targets, the intercept diffuse until its first row. Measured 2026-11-24:
/// within 2.3e-15 of it (over `1 + |value|`) at lags 1, 7 and 40, against a
/// bound of 1e-13.
#[test]
fn the_ring_is_the_whole_prefix_smoother_on_a_random_walk() {
    for lag in [1, 7, 40] {
        let rows = stream(400, 2, 1, 496 + lag as u64, false);
        let model = Kalman::new(base(2, 1))
            .unwrap()
            .with_smoother(lag, true)
            .unwrap();
        let r = run(model, &rows, &ids(rows.len()));
        assert!(
            r.emitted.len() > 250,
            "lag {lag}: {} emitted",
            r.emitted.len()
        );
        let (err, _) = check(&r, lag, &|_, _| identity(3));
        eprintln!("random walk, lag {lag}: {err:.1e}");
        assert!(err < 1e-13, "lag {lag}: {err:e}");
    }
}

/// Unstandardized with reverting slots, `F = diag(2^(−D/r_i))` between two
/// observations `D` apart, and the means alone (no variances kept).
/// Measured 2026-11-24: within 7.2e-16, against a bound of 1e-13.
#[test]
fn the_ring_is_the_whole_prefix_smoother_with_reversion() {
    let lag = 12;
    let rows = stream(300, 2, 1, 4961, false);
    let cfg = KalmanCfg {
        revert_half_life: vec![f64::INFINITY, 20.0, 50.0],
        ..base(2, 1)
    };
    let model = Kalman::new(cfg.clone())
        .unwrap()
        .with_smoother(lag, false)
        .unwrap();
    let r = run(model, &rows, &ids(rows.len()));
    // The clock between observation `n` and `n + 1`, from the rows.
    let obs: Vec<usize> = (0..rows.len())
        .filter(|&i| rows[i].w > 0.0 && rows[i].y[0].is_some())
        .collect();
    let f = |_: usize, n: usize| {
        let d: f64 = rows[obs[n] + 1..=obs[n + 1]].iter().map(|r| r.d).sum();
        let mut out = identity(3);
        for i in 0..3 {
            out[i * 3 + i] = cfg.phi(i, d);
        }
        out
    };
    let (err, _) = check(&r, lag, &f);
    eprintln!("reversion: {err:.1e}");
    assert!(err < 1e-13, "{err:e}");
}

/// Standardized, the features' spread growing so the filter re-maps inside
/// the window: in the caller's units a re-map is no change of the
/// coefficients, so the written smoother runs with `F = I` there, and the
/// ring, which carries the pending pair across each re-map, is it.
/// Measured 2026-11-24: 7 re-maps, within 8.0e-15, against a bound of
/// 1e-12.
#[test]
fn the_ring_is_the_whole_prefix_smoother_across_re_maps() {
    let lag = 25;
    let rows = stream(500, 3, 1, 4962, true);
    let cfg = KalmanCfg {
        standardize: true,
        ..base(3, 1)
    };
    let model = Kalman::new(cfg).unwrap().with_smoother(lag, true).unwrap();
    let r = run(model, &rows, &ids(rows.len()));
    let log = r.model.smooth_log();
    let remaps = log
        .windows(2)
        .filter(|w| w[0].anchor[3 * 4 - 1] != w[1].anchor[3 * 4 - 1])
        .count();
    assert!(remaps >= 3, "{remaps} re-maps");
    let (err, _) = check(&r, lag, &|_, _| identity(4));
    eprintln!("re-maps ({remaps}): {err:.1e}");
    assert!(err < 1e-12, "{err:e}");
}

/// `share_p`: one covariance, each target's means; a target null on the
/// first rows is not described until it is observed ([`SMOOTH_UNSEEN`]).
/// Measured 2026-11-24: within 1.6e-13, against a bound of 1e-11.
#[test]
fn the_ring_is_the_whole_prefix_smoother_under_share_p() {
    let lag = 9;
    let mut rows = stream(300, 2, 2, 4963, false);
    for row in rows.iter_mut().take(30) {
        row.y[1] = None;
    }
    let cfg = KalmanCfg {
        share_p: true,
        standardize: true,
        ..base(2, 2)
    };
    let model = Kalman::new(cfg).unwrap().with_smoother(lag, true).unwrap();
    let r = run(model, &rows, &ids(rows.len()));
    let (err, held) = check(&r, lag, &|_, _| identity(3));
    eprintln!("share_p: {err:.1e}");
    assert!(err < 1e-11, "{err:e}");
    assert!(
        held > 0,
        "the second target's first observations are not described"
    );
    let first = &r.emitted[0].1;
    assert_eq!(first.held[1], SMOOTH_UNSEEN);
    assert!(first.coef[3..6].iter().all(|v| v.is_nan()));
}

/// Every prediction and every number the filter keeps is the same with
/// the smoother on or off, to the bit: it reads the filter and writes only
/// its own slots.
#[test]
fn the_filter_is_the_same_to_the_bit_with_the_smoother() {
    let cfgs = [
        base(2, 1),
        KalmanCfg {
            standardize: true,
            ..base(3, 1)
        },
        KalmanCfg {
            standardize: true,
            revert_half_life: vec![f64::INFINITY, 15.0, 40.0, 90.0],
            ..base(3, 1)
        },
        KalmanCfg {
            share_p: true,
            standardize: true,
            ..base(2, 2)
        },
        KalmanCfg {
            obs_var: Some(0.3),
            standardize: true,
            ..base(2, 2)
        },
    ];
    for (c, cfg) in cfgs.into_iter().enumerate() {
        let rows = stream(400, cfg.n_features, cfg.n_targets, 500 + c as u64, c == 1);
        let plain = run(Kalman::new(cfg.clone()).unwrap(), &rows, &ids(400));
        let smoothed = Kalman::new(cfg)
            .unwrap()
            .with_smoother(10, c % 2 == 0)
            .unwrap();
        let smoothed = run(smoothed, &rows, &ids(400));
        assert_eq!(plain.preds, smoothed.preds, "case {c}");
        let mut bare = smoothed.model.clone();
        bare.smoother = None;
        assert_eq!(
            rmp_serde::to_vec_named(&plain.model).unwrap(),
            rmp_serde::to_vec_named(&bare).unwrap(),
            "case {c}"
        );
        assert!(!smoothed.emitted.is_empty());
    }
}

/// A row of weight 0, whatever its target (1e300 or null), takes no slot
/// and counts in no lag: the stream with such rows emits on every other row
/// what the stream with them dropped, their clock carried into the next
/// row, emits, and nothing on them; one such row first. Unstandardized and
/// with no decay to split, to the bit; standardized, the moments' decay
/// over a split gap is a product of two factors where the dropped stream
/// takes one, and the filter itself moves by rounding
/// (`tests/kalman_clock.rs`, `a_zero_weight_row_inside_a_gap_moves_nothing`):
/// within 1e-12 there.
#[test]
fn a_row_of_weight_0_is_no_observation() {
    for (standardize, decay) in [(false, f64::INFINITY), (true, 100.0)] {
        let cfg = KalmanCfg {
            standardize,
            decay: Decay::Halflife(decay),
            ..base(2, 1)
        };
        let mut rows = stream(300, 2, 1, 4964, false);
        rows[0].w = 0.0;
        rows[0].y = vec![Some(1e300)];
        let stamps = ids(rows.len());
        let mut kept = vec![];
        let mut kept_stamps = vec![];
        let mut carry = 0.0;
        for (r, row) in rows.iter().enumerate() {
            if row.w == 0.0 {
                carry += row.d;
                continue;
            }
            let mut row = row.clone();
            row.d += carry;
            carry = 0.0;
            kept.push(row);
            kept_stamps.push(stamps[r]);
        }
        assert!(kept.len() < rows.len() - 10);
        let all = run(
            Kalman::new(cfg.clone())
                .unwrap()
                .with_smoother(5, true)
                .unwrap(),
            &rows,
            &stamps,
        );
        let dropped = run(
            Kalman::new(cfg).unwrap().with_smoother(5, true).unwrap(),
            &kept,
            &kept_stamps,
        );
        assert_eq!(all.emitted.len(), dropped.emitted.len());
        for ((ra, a, _), (rd, d, _)) in all.emitted.iter().zip(&dropped.emitted) {
            assert!(rows[*ra].w > 0.0);
            assert_eq!(stamps[*ra], kept_stamps[*rd]);
            assert_eq!(a.stamp, d.stamp);
            assert_eq!(a.held, d.held);
            for (u, v) in a.coef.iter().chain(&a.var).zip(d.coef.iter().chain(&d.var)) {
                if u.is_nan() || v.is_nan() {
                    assert!(u.is_nan() && v.is_nan(), "{u} against {v}");
                } else if standardize {
                    assert!((u - v).abs() <= 1e-12 * (1.0 + v.abs()), "{u} against {v}");
                } else {
                    assert_eq!(u.to_bits(), v.to_bits(), "{u} against {v}");
                }
            }
        }
    }
}

/// A state saved with the ring part full, loaded, goes on to emit what the
/// unbroken run emits, to the bit, and re-saves byte for byte.
#[test]
fn a_state_saved_mid_ring_goes_on_to_the_bit() {
    let cfg = KalmanCfg {
        standardize: true,
        ..base(3, 1)
    };
    let rows = stream(400, 3, 1, 4965, true);
    let stamps = ids(400);
    let whole = run(
        Kalman::new(cfg.clone())
            .unwrap()
            .with_smoother(30, true)
            .unwrap(),
        &rows,
        &stamps,
    );
    let first = run(
        Kalman::new(cfg).unwrap().with_smoother(30, true).unwrap(),
        &rows[..217],
        &stamps[..217],
    );
    let bytes = rmp_serde::to_vec_named(&first.model.state()).unwrap();
    let state: State = rmp_serde::from_slice(&bytes).unwrap();
    let loaded = Kalman::restore(&state).unwrap();
    assert_eq!(rmp_serde::to_vec_named(&loaded.state()).unwrap(), bytes);
    let rest = run(loaded, &rows[217..], &stamps[217..]);
    let joined: Vec<_> = first
        .emitted
        .iter()
        .map(|(r, s, _)| (*r, s))
        .chain(rest.emitted.iter().map(|(r, s, _)| (r + 217, s)))
        .collect();
    assert_eq!(joined.len(), whole.emitted.len());
    for ((ra, a), (rb, b, _)) in joined.iter().zip(&whole.emitted) {
        assert_eq!(ra, rb);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.coef), bits(&b.coef));
        assert_eq!(bits(&a.var), bits(&b.var));
        assert_eq!(a.stamp, b.stamp);
    }
}

/// A re-map refused -- a feature's spread moving by more than the re-map
/// carries in one row -- cuts the chain: the observations before it are
/// emitted as the stream stopped at that row would leave them, smoothed
/// with what came before the break, the coefficients to the bit.
#[test]
fn a_refused_re_map_cuts_the_chain() {
    let cfg = KalmanCfg {
        standardize: true,
        ..base(2, 1)
    };
    let mut rows = stream(200, 2, 1, 4966, false);
    for row in &mut rows {
        row.w = 1.0;
        row.y = vec![row.y[0].filter(|y| y.abs() < 1e10).or(Some(0.5))];
    }
    let jump = 120;
    rows[jump].x[0] = 1e7;
    let lag = 10;
    let stamps = ids(rows.len());
    let whole = run(
        Kalman::new(cfg.clone())
            .unwrap()
            .with_smoother(lag, true)
            .unwrap(),
        &rows,
        &stamps,
    );
    let cut = run(
        Kalman::new(cfg).unwrap().with_smoother(lag, true).unwrap(),
        &rows[..=jump],
        &stamps[..=jump],
    );
    let tail = cut.model.smoothed_tail().unwrap();
    assert_eq!(tail.len(), lag);
    for row in &tail {
        let (_, out, _) = whole
            .emitted
            .iter()
            .find(|(_, o, _)| o.stamp[0] == Some(row.stamp))
            .expect("emitted after the break");
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        // The variances read the noise as it stands, which the rows after
        // the break move; the coefficients read nothing after it.
        assert_eq!(bits(&row.coef), bits(&out.coef), "{:?}", row.stamp);
    }
    // Without the break the rows after it move those values.
    let mut smooth_rows = rows.clone();
    smooth_rows[jump].x[0] = 0.5;
    let unbroken = run(
        Kalman::new(KalmanCfg {
            standardize: true,
            ..base(2, 1)
        })
        .unwrap()
        .with_smoother(lag, true)
        .unwrap(),
        &smooth_rows[..=jump],
        &stamps[..=jump],
    );
    let moved = unbroken.model.smoothed_tail().unwrap();
    let (_, later, _) = whole
        .emitted
        .iter()
        .find(|(_, o, _)| o.stamp[0] == Some(moved[0].stamp))
        .unwrap();
    assert_ne!(
        moved[0]
            .coef
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        later.coef.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

/// A covariance sized again after losing its definiteness cuts its chain as
/// a refused re-map does.
#[test]
fn a_covariance_sized_again_cuts_the_chain() {
    let rows = stream(120, 2, 1, 4967, false);
    let lag = 6;
    let model = Kalman::new(base(2, 1))
        .unwrap()
        .with_smoother(lag, false)
        .unwrap();
    let mut first = run(model, &rows[..80], &ids(80));
    let before = first.model.smoothed_tail().unwrap();
    first.model.size_again(0);
    let stamps = ids(120);
    let rest = run(first.model, &rows[80..], &stamps[80..]);
    for row in &before {
        let (_, out, _) = rest
            .emitted
            .iter()
            .find(|(_, o, _)| o.stamp[0] == Some(row.stamp))
            .expect("emitted after the break");
        assert_eq!(row.coef, out.coef[..3]);
    }
}

/// `min_weight` withholds a row's prediction, and so the smoothed value
/// that describes it: emitted on its lag's row, NaN, with the reason.
#[test]
fn the_filters_gate_withholds_the_smoothed_value() {
    let cfg = KalmanCfg {
        min_weight: 20.0,
        decay: Decay::Halflife(f64::INFINITY),
        ..base(2, 1)
    };
    let rows = stream(100, 2, 1, 4968, false);
    let r = run(
        Kalman::new(cfg).unwrap().with_smoother(4, true).unwrap(),
        &rows,
        &ids(100),
    );
    let first = &r.emitted[0].1;
    assert_eq!(first.held[0], SMOOTH_MIN_WEIGHT);
    assert!(first.coef.iter().chain(&first.var).all(|v| v.is_nan()));
    let last = &r.emitted.last().unwrap().1;
    assert_eq!(last.held[0], SMOOTH_STOOD);
    assert!(last.coef.iter().all(|v| v.is_finite()));
}

/// A lag of 0, the square root's form either way round, and a saved
/// smoother of the wrong shape are refused, each saying so.
#[test]
fn the_smoothers_refusals() {
    let e = Kalman::new(base(2, 1))
        .unwrap()
        .with_smoother(0, false)
        .unwrap_err();
    assert!(e.contains("smooth_lag must be at least 1"), "{e}");
    let e = Kalman::new(base(2, 1))
        .unwrap()
        .with_covariance_form(CovarianceForm::SquareRoot)
        .unwrap()
        .with_smoother(3, false)
        .unwrap_err();
    assert!(e.contains("square_root"), "{e}");
    let e = Kalman::new(base(2, 1))
        .unwrap()
        .with_smoother(3, false)
        .unwrap()
        .with_covariance_form(CovarianceForm::SquareRoot)
        .unwrap_err();
    assert!(e.contains("square_root"), "{e}");
    let rows = stream(40, 2, 1, 4969, false);
    let r = run(
        Kalman::new(base(2, 1))
            .unwrap()
            .with_smoother(5, true)
            .unwrap(),
        &rows,
        &ids(40),
    );
    let bytes = rmp_serde::to_vec_named(&r.model).unwrap();
    let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
    let rmpv::Value::Map(fields) = &mut v else {
        panic!("a map")
    };
    let (_, sm) = fields
        .iter_mut()
        .find(|(k, _)| k.as_str() == Some("smoother"))
        .expect("a smoother written");
    let rmpv::Value::Map(sf) = sm else {
        panic!("a map")
    };
    let (_, lag) = sf
        .iter_mut()
        .find(|(k, _)| k.as_str() == Some("lag"))
        .unwrap();
    *lag = rmpv::Value::from(2u64);
    let bytes = {
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &v).unwrap();
        out
    };
    let e = rmp_serde::from_slice::<Kalman>(&bytes)
        .unwrap_err()
        .to_string();
    assert!(e.contains("smoother has the wrong shape"), "{e}");
}
