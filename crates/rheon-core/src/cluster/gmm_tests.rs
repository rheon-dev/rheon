//! The mixture's recursion and its gated start: the read against the
//! mixture density written out, the online step against batch EM on the
//! same rows, the start's three conditions and its fixed order, the
//! buffer's bound, zero-weight rows, and a mid-stream save and load
//! continuing to the bit.

use super::*;

/// A seeded stream from `K` separated Gaussian components in two
/// dimensions, means `(4 j, 3 j)`, the covariances of the study's pair.
fn stream(n: usize, kk: usize, seed: u64) -> Vec<[f64; 2]> {
    let mut rng = SplitMix64::new(seed);
    let mut normal = || {
        let (u, v) = (rng.uniform().max(1e-300), rng.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let covs: [[f64; 3]; 3] = [[1.0, 0.5, 1.0], [1.5, -0.4, 0.8], [0.7, 0.1, 0.6]];
    (0..n)
        .map(|t| {
            // Component weights 0.35 / 0.65 (two), or by the row's index.
            let j = if kk == 1 {
                0
            } else if kk == 2 {
                usize::from((t * 7919 % 100) >= 35)
            } else {
                t * 7919 % kk
            };
            let [a, b, c] = covs[j % 3];
            let (l11, l21) = (a.sqrt(), b / a.sqrt());
            let l22 = (c - l21 * l21).sqrt();
            let (z1, z2) = (normal(), normal());
            [
                4.0 * j as f64 + l11 * z1,
                3.0 * j as f64 + l21 * z1 + l22 * z2,
            ]
        })
        .collect()
}

const ALL: [GmmCovariance; 4] = [
    GmmCovariance::Full,
    GmmCovariance::Diag,
    GmmCovariance::Spherical,
    GmmCovariance::Tied,
];

fn feed(g: &mut Gmm, rows: &[[f64; 2]], lam: f64) {
    for x in rows {
        g.learn(x, 1.0, lam);
    }
}

#[test]
fn a_separated_stream_starts_and_orders_its_components_by_weight() {
    let mut g = Gmm::new(GmmCfg::new(2, 2, GmmCovariance::Full)).unwrap();
    assert_eq!(g.wait(), Some(GmmWait::WarmUp));
    let rows = stream(4000, 2, 1);
    let mut started_at = None;
    for (t, x) in rows.iter().enumerate() {
        g.learn(x, 1.0, 1.0);
        if started_at.is_none() && g.started() {
            started_at = Some(t + 1);
        }
    }
    let at = started_at.expect("a separated pair starts");
    // The start is tried at doublings of the buffer, so it fires at one.
    assert!(at.is_power_of_two(), "started at {at}");
    let p = g.parameters().unwrap();
    // Largest first: 0.65 at (4, 3), then 0.35 at the origin.
    assert!(p.weights[0] > p.weights[1]);
    assert!(
        (p.means[0] - 4.0).abs() < 0.2 && (p.means[1] - 3.0).abs() < 0.2,
        "{p:?}"
    );
    assert!(p.means[2].abs() < 0.2 && p.means[3].abs() < 0.2, "{p:?}");
    assert!((p.weights[0] - 0.65).abs() < 0.03, "{p:?}");
}

#[test]
fn the_coefficients_are_each_components_weight_mean_and_packed_covariance() {
    for cov in ALL {
        let mut g = Gmm::new(GmmCfg::new(2, 2, cov)).unwrap();
        assert_eq!(g.coefficients(), None);
        feed(&mut g, &stream(2000, 2, 1), 1.0);
        let (p, c) = (g.parameters().unwrap(), g.coefficients().unwrap());
        assert_eq!(c.len(), 2);
        for (j, row) in c.iter().enumerate() {
            let want: Vec<f64> = match cov {
                GmmCovariance::Full => {
                    let s = &p.covs[4 * j..4 * j + 4];
                    vec![
                        p.weights[j],
                        p.means[2 * j],
                        p.means[2 * j + 1],
                        s[0],
                        s[1],
                        s[3],
                    ]
                }
                GmmCovariance::Diag => {
                    let s = &p.covs[2 * j..2 * j + 2];
                    vec![p.weights[j], p.means[2 * j], p.means[2 * j + 1], s[0], s[1]]
                }
                GmmCovariance::Spherical => {
                    vec![p.weights[j], p.means[2 * j], p.means[2 * j + 1], p.covs[j]]
                }
                GmmCovariance::Tied => {
                    let s = &p.covs[..4];
                    vec![
                        p.weights[j],
                        p.means[2 * j],
                        p.means[2 * j + 1],
                        s[0],
                        s[1],
                        s[3],
                    ]
                }
            };
            assert_eq!(row, &want, "{cov:?} component {j}");
        }
    }
}

#[test]
fn one_cluster_asked_for_two_never_starts() {
    let mut g = Gmm::new(GmmCfg::new(2, 2, GmmCovariance::Full)).unwrap();
    feed(&mut g, &stream(3000, 1, 2), 1.0);
    assert!(!g.started());
    assert_eq!(g.wait(), Some(GmmWait::NotSeparated));
}

#[test]
fn one_component_starts_once_its_fit_stops_moving() {
    let mut g = Gmm::new(GmmCfg::new(2, 1, GmmCovariance::Diag)).unwrap();
    feed(&mut g, &stream(500, 1, 3), 1.0);
    assert!(g.started());
}

#[test]
fn a_buffer_at_its_bound_is_withheld_and_never_forced_to_start() {
    let mut cfg = GmmCfg::new(2, 2, GmmCovariance::Full);
    cfg.max_state_bytes = 24 * 10; // ten rows of two features and a weight
    let mut g = Gmm::new(cfg).unwrap();
    feed(&mut g, &stream(200, 2, 4), 1.0);
    assert!(!g.started());
    assert_eq!(g.wait(), Some(GmmWait::OverCap));
    assert!(g.buffer_bytes() <= 240);
}

/// The mixture density of a 2-d row, written out with the 2 × 2 inverse
/// and determinant.
fn density(p: &Fit, x: &[f64; 2]) -> (Vec<f64>, f64) {
    let kk = p.weights.len();
    let terms: Vec<f64> = (0..kk)
        .map(|j| {
            let c = &p.covs[j * 4..(j + 1) * 4];
            let det = c[0] * c[3] - c[1] * c[2];
            let (d0, d1) = (x[0] - p.means[j * 2], x[1] - p.means[j * 2 + 1]);
            let q = (c[3] * d0 * d0 - 2.0 * c[1] * d0 * d1 + c[0] * d1 * d1) / det;
            p.weights[j] * (-0.5 * q).exp() / (std::f64::consts::TAU * det.sqrt())
        })
        .collect();
    let f: f64 = terms.iter().sum();
    (terms.iter().map(|t| t / f).collect(), f.ln())
}

#[test]
fn the_read_is_the_mixture_density_and_its_responsibilities() {
    let mut g = Gmm::new(GmmCfg::new(2, 2, GmmCovariance::Full)).unwrap();
    feed(&mut g, &stream(3000, 2, 5), 1.0);
    let p = g.parameters().unwrap();
    for x in [[0.3, -0.2], [2.0, 1.5], [4.5, 2.0], [9.0, -7.0]] {
        let r = g.read(&x).unwrap();
        let (resp, logf) = density(&p, &x);
        assert!(
            (r.log_density - logf).abs() < 1e-12 * logf.abs().max(1.0),
            "{x:?}"
        );
        for (a, b) in r.resp.iter().zip(&resp) {
            assert!((a - b).abs() < 1e-12, "{x:?}");
        }
        assert!((r.resp.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert_eq!(r.cluster, usize::from(r.resp[1] > r.resp[0]));
    }
}

#[test]
fn the_online_fit_is_batch_em_on_the_same_rows_within_its_error() {
    for cov in ALL {
        let rows = stream(20000, 2, 6);
        let mut g = Gmm::new(GmmCfg::new(2, 2, cov)).unwrap();
        feed(&mut g, &rows, 1.0);
        let p = g.parameters().unwrap();
        let xs: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let ws = vec![1.0; rows.len()];
        let mut cfg = g.cfg().clone();
        cfg.tol = 1e-10;
        cfg.max_iter = 1000;
        let b = best_fit(&cfg, &xs, &ws, 2, None).unwrap();
        // Match by the first mean coordinate; the batch fit is not ordered.
        let order: Vec<usize> = if (b.means[0] - p.means[0]).abs() < 1.0 {
            vec![0, 1]
        } else {
            vec![1, 0]
        };
        for (j, &bj) in order.iter().enumerate() {
            // Within a tenth of a unit on means of unit-variance clusters
            // at 20,000 rows: a separated pair, where one pass of online EM
            // lands within one batch standard error of batch EM.
            for i in 0..2 {
                assert!(
                    (p.means[j * 2 + i] - b.means[bj * 2 + i]).abs() < 0.05,
                    "{cov:?} {p:?} {b:?}"
                );
            }
            assert!((p.weights[j] - b.weights[bj]).abs() < 0.01, "{cov:?}");
        }
        // The covariances too, each read as the type keeps it (`reg_covar`
        // is in the online read and not in the batch fit's moments).
        let per = cov.per(2);
        let (pc, bc) = (p.covs.len(), b.covs.len());
        assert_eq!(pc, bc, "{cov:?}");
        for (j, &bj) in order.iter().enumerate() {
            let at = |o: usize| if per == 0 { 0 } else { o * per };
            for i in 0..per.max(if per == 0 { 4 } else { 0 }) {
                let (a, c) = (p.covs[at(j) + i], b.covs[at(bj) + i]);
                assert!((a - c).abs() < 0.06, "{cov:?} {a} {c}");
            }
        }
    }
}

#[test]
fn a_zero_weight_row_decays_the_weights_and_moves_no_mean_or_covariance() {
    let mut g = Gmm::new(GmmCfg::new(2, 2, GmmCovariance::Full)).unwrap();
    feed(&mut g, &stream(3000, 2, 7), 0.999);
    let before = g.parameters().unwrap();
    let n = g.n_eff();
    g.learn(&[100.0, -50.0], 0.0, 0.999);
    let after = g.parameters().unwrap();
    assert_eq!(before.means, after.means);
    assert_eq!(before.covs, after.covs);
    assert!((g.n_eff() - 0.999 * n).abs() < 1e-9 * n);
    // A zero-weight first row of an empty model leaves nothing.
    let mut e = Gmm::new(GmmCfg::new(2, 2, GmmCovariance::Diag)).unwrap();
    e.learn(&[1.0, 2.0], 0.0, 0.9);
    assert_eq!(e.n_eff(), 0.0);
    assert_eq!(e.buffer_bytes(), 0);
}

#[test]
fn a_mid_stream_save_and_load_continues_to_the_bit() {
    for cov in ALL {
        let rows = stream(3000, 2, 8);
        for cut in [3, 600, 2000] {
            let mut a = Gmm::new(GmmCfg::new(2, 2, cov)).unwrap();
            feed(&mut a, &rows[..cut], 0.9995);
            let saved = rmp_serde::to_vec_named(&a).unwrap();
            let mut b: Gmm = rmp_serde::from_slice(&saved).unwrap();
            assert_eq!(rmp_serde::to_vec_named(&b).unwrap(), saved);
            for x in &rows[cut..] {
                let (ra, rb) = (a.read(x), b.read(x));
                assert_eq!(
                    ra.as_ref().map(|r| (r.log_density.to_bits(), r.cluster)),
                    rb.as_ref().map(|r| (r.log_density.to_bits(), r.cluster)),
                    "{cov:?} cut {cut}"
                );
                a.learn(x, 1.0, 0.9995);
                b.learn(x, 1.0, 0.9995);
            }
            assert_eq!(
                rmp_serde::to_vec_named(&a).unwrap(),
                rmp_serde::to_vec_named(&b).unwrap()
            );
        }
    }
}

#[test]
fn a_far_row_gets_a_finite_read() {
    let mut g = Gmm::new(GmmCfg::new(2, 2, GmmCovariance::Full)).unwrap();
    feed(&mut g, &stream(3000, 2, 9), 1.0);
    let r = g.read(&[1e6, -1e6]).unwrap();
    assert!(r.log_density.is_finite() && r.resp.iter().all(|v| v.is_finite()));
}

#[test]
fn refusals_name_the_setting() {
    let bad = |f: fn(&mut GmmCfg)| {
        let mut c = GmmCfg::new(2, 2, GmmCovariance::Full);
        f(&mut c);
        Gmm::new(c).unwrap_err()
    };
    assert!(bad(|c| c.n_components = 0).contains("n_components"));
    assert!(bad(|c| c.reg_covar = 0.0).contains("reg_covar"));
    assert!(bad(|c| c.delta_bic = -1.0).contains("delta_bic"));
    assert!(bad(|c| c.tol = f64::NAN).contains("tol"));
}

#[test]
fn spherical_reads_the_mean_variance_and_tied_one_shared_covariance() {
    let rows = stream(4000, 2, 11);
    let xs: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let ws = vec![1.0; rows.len()];
    // The M-steps, from fixed responsibilities, against their definitions:
    // spherical's variance is the mean of the diagonal's, and tied's
    // covariance is the components' scatter summed over the total mass.
    let resp: Vec<f64> = rows
        .iter()
        .flat_map(|r| if r[0] > 2.0 { [0.9, 0.1] } else { [0.2, 0.8] })
        .collect();
    let cfg = |c| GmmCfg::new(2, 2, c);
    let full = m_step(&cfg(GmmCovariance::Full), &xs, &ws, &resp, 2).unwrap();
    let diag = m_step(&cfg(GmmCovariance::Diag), &xs, &ws, &resp, 2).unwrap();
    let sph = m_step(&cfg(GmmCovariance::Spherical), &xs, &ws, &resp, 2).unwrap();
    let tied = m_step(&cfg(GmmCovariance::Tied), &xs, &ws, &resp, 2).unwrap();
    for j in 0..2 {
        let want = (diag.covs[2 * j] + diag.covs[2 * j + 1]) / 2.0;
        assert!((sph.covs[j] - want).abs() < 1e-12 * want, "{j}");
    }
    let total: f64 = full.mass.iter().sum();
    for i in 0..4 {
        let want = (full.mass[0] * full.covs[i] + full.mass[1] * full.covs[4 + i]) / total;
        assert!((tied.covs[i] - want).abs() < 1e-12, "{i}");
    }
    // Online, tied's total form is the same covariance as the components'
    // scatter: `C_total − Σ π_j (μ_j − m)(μ_j − m)ᵀ`, symmetric and definite.
    let mut g = Gmm::new(cfg(GmmCovariance::Tied)).unwrap();
    feed(&mut g, &rows, 0.999);
    let p = g.parameters().unwrap();
    assert_eq!(p.covs.len(), 4);
    assert_eq!(p.covs[1], p.covs[2]);
    assert!(strictly_pd(&p.covs, 2));
    let c = g.coefficients().unwrap();
    assert_eq!(
        c[0][3..],
        c[1][3..],
        "every row carries the shared covariance"
    );
}

#[test]
fn every_type_starts_on_a_separated_pair_and_never_on_one_cluster() {
    for cov in ALL {
        let mut g = Gmm::new(GmmCfg::new(2, 2, cov)).unwrap();
        feed(&mut g, &stream(4000, 2, 12), 1.0);
        assert!(g.started(), "{cov:?}");
        // The one cluster is correlated, which a diagonal or spherical
        // mixture fits better with two components: those two may start.
        if matches!(cov, GmmCovariance::Full | GmmCovariance::Tied) {
            let mut one = Gmm::new(GmmCfg::new(2, 2, cov)).unwrap();
            feed(&mut one, &stream(3000, 1, 13), 1.0);
            assert!(!one.started(), "{cov:?}");
        }
    }
}
