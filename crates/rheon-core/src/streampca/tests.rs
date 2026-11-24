//! `StreamPca`'s unit tests: the incremental SVD against faer's
//! self-adjoint eigensolver on the decayed, weighted covariance written out
//! longhand (`crate::oracle`, which this model does not call), the two
//! stochastic recursions' convergence to the same eigenvectors on a
//! separated spectrum, the eigengap gate on both sides of its threshold,
//! zero-weight rows, the sign rule, the state and the refusals.

use super::*;
use crate::{Decay, OnlineModel};

/// A seeded uniform in `(0, 1)` (xorshift64*).
fn uniform(s: &mut u64) -> f64 {
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
    (v as f64 + 0.5) / (1u64 << 53) as f64
}

/// A seeded standard normal (Box-Muller).
fn normal(s: &mut u64) -> f64 {
    let (u1, u2) = (uniform(s), uniform(s));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A random orthonormal `k x k` basis, rows, by Gram-Schmidt on normals.
fn basis(k: usize, s: &mut u64) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    while out.len() < k {
        let mut v: Vec<f64> = (0..k).map(|_| normal(s)).collect();
        for b in &out {
            let p: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
            for (vi, bi) in v.iter_mut().zip(b) {
                *vi -= p * bi;
            }
        }
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        out.push(v.into_iter().map(|x| x / n).collect());
    }
    out
}

/// Rows `mean + sum_j sqrt(spectrum_j) z_j b_j`: a Gaussian stream whose
/// covariance has eigenvalues `spectrum` along the rows of a random basis.
struct Stream {
    basis: Vec<Vec<f64>>,
    sd: Vec<f64>,
    mean: Vec<f64>,
    seed: u64,
}

impl Stream {
    fn new(spectrum: &[f64], seed: u64) -> Self {
        let mut s = seed;
        let k = spectrum.len();
        let basis = basis(k, &mut s);
        let mean = (0..k).map(|i| 5.0 + i as f64 * 0.5).collect();
        Self {
            basis,
            sd: spectrum.iter().map(|l| l.sqrt()).collect(),
            mean,
            seed: s,
        }
    }

    fn row(&mut self) -> Vec<f64> {
        let mut x = self.mean.clone();
        for (b, sd) in self.basis.iter().zip(&self.sd) {
            let z = normal(&mut self.seed) * sd;
            for (xi, bi) in x.iter_mut().zip(b) {
                *xi += z * bi;
            }
        }
        x
    }
}

fn cfg(k: usize, r: usize, method: PcaMethod, decay: Decay) -> StreamPcaCfg {
    StreamPcaCfg {
        n_features: k,
        n_components: r,
        method,
        decay,
        min_weight: 0.0,
        learning_rate: None,
        max_state_bytes: None,
    }
}

/// The decayed, weighted covariance of `rows` in the mean form, written out
/// longhand: each row's weight now `u_i = w_i prod lam` over the rows after
/// it, `m = sum u x / U`, `C = sum u (x - m)(x - m)' / U`, row-major.
fn covariance(rows: &[(Vec<f64>, f64)], lam: f64) -> (Vec<f64>, f64) {
    let k = rows[0].0.len();
    let n = rows.len();
    let u: Vec<f64> = rows
        .iter()
        .enumerate()
        .map(|(i, (_, w))| w * lam.powi((n - 1 - i) as i32))
        .collect();
    let total: f64 = u.iter().sum();
    let mut m = vec![0.0; k];
    for ((x, _), ui) in rows.iter().zip(&u) {
        for (mi, xi) in m.iter_mut().zip(x) {
            *mi += ui * xi / total;
        }
    }
    let mut c = vec![0.0; k * k];
    for ((x, _), ui) in rows.iter().zip(&u) {
        for i in 0..k {
            for j in 0..k {
                c[i * k + j] += ui * (x[i] - m[i]) * (x[j] - m[j]) / total;
            }
        }
    }
    (c, total)
}

/// The largest `1 - |cos|` between each kept direction and the oracle's
/// eigenvector of the same rank, over the first `r`.
fn worst_alignment(rows: &[f64], k: usize, r: usize, vectors: &[Vec<f64>]) -> f64 {
    let n = vectors.len();
    (0..r)
        .map(|j| {
            let v = &vectors[n - 1 - j];
            let c: f64 = rows[j * k..(j + 1) * k]
                .iter()
                .zip(v)
                .map(|(a, b)| a * b)
                .sum();
            1.0 - c.abs()
        })
        .fold(0.0, f64::max)
}

/// At `n_components = k` the incremental SVD keeps every direction, so it
/// is the eigendecomposition of the decayed, weighted covariance itself:
/// faer's eigensolver on the covariance formed longhand, to rounding, under
/// decay and unequal weights with zero weights among them.
#[test]
fn the_full_svd_is_the_eigendecomposition_of_the_weighted_covariance() {
    for (lam, seed) in [(1.0, 1u64), (0.99, 2), (0.9, 3)] {
        let k = 4;
        let mut m = StreamPca::new(cfg(k, k, PcaMethod::Svd, Decay::Lam(lam))).unwrap();
        let mut stream = Stream::new(&[9.0, 4.0, 1.5, 0.5], seed);
        let mut s = seed * 77 + 1;
        let mut rows = Vec::new();
        for i in 0..400 {
            let x = stream.row();
            let w = if i % 10 == 3 {
                0.0
            } else {
                0.5 + uniform(&mut s)
            };
            m.step(&x, &[], 1.0, w);
            rows.push((x, w));
        }
        let (c, total) = covariance(&rows, lam);
        assert!((m.n_eff() - total).abs() <= 1e-12 * total, "lam {lam}");
        let (values, vectors) = crate::oracle::sym_eigen(&c);
        let (dirs, vals) = m.directions();
        for j in 0..k {
            let want = values[k - 1 - j];
            assert!(
                (vals[j] - want).abs() <= 1e-12 * values[k - 1],
                "lam {lam}: component {j} variance {} against {want}",
                vals[j]
            );
        }
        let worst = worst_alignment(dirs, k, k, &vectors);
        assert!(worst <= 1e-12, "lam {lam}: 1 - |cos| {worst:e}");
        // The trace is the covariance's, exactly as the diagonal recursion
        // keeps it.
        let tr: f64 = (0..k).map(|i| c[i * k + i]).sum();
        assert!((m.trace() - tr).abs() <= 1e-12 * tr);
    }
}

/// Truncated below `k`, the incremental SVD is still exact while the data's
/// rank is at most the directions it keeps: a 5-column stream on a plane
/// (rank 2), two components emitted and three kept.
#[test]
fn a_truncated_svd_is_exact_on_data_of_lower_rank() {
    let k = 5;
    let lam = 0.995;
    let mut m = StreamPca::new(cfg(k, 2, PcaMethod::Svd, Decay::Lam(lam))).unwrap();
    let mut stream = Stream::new(&[6.0, 2.0, 0.0, 0.0, 0.0], 11);
    let mut rows = Vec::new();
    for _ in 0..300 {
        let x = stream.row();
        m.step(&x, &[], 1.0, 1.0);
        rows.push((x, 1.0));
    }
    let (c, _) = covariance(&rows, lam);
    let (values, vectors) = crate::oracle::sym_eigen(&c);
    let (dirs, vals) = m.directions();
    for j in 0..2 {
        assert!((vals[j] - values[k - 1 - j]).abs() <= 1e-12 * values[k - 1]);
    }
    assert!(worst_alignment(dirs, k, 2, &vectors) <= 1e-12);
    // The third kept direction is the rank's rounding, which the gate
    // reads as no direction.
    assert!(vals[2] <= 1e-12 * vals[0], "{vals:?}");
    assert!(!m.separated(2));
}

/// The kept directions are orthonormal to rounding after a long stream: the
/// rotation makes them orthogonal afresh on every row, so nothing drifts.
#[test]
fn the_svd_directions_stay_orthonormal() {
    let k = 6;
    let mut m = StreamPca::new(cfg(k, 3, PcaMethod::Svd, Decay::Lam(0.999))).unwrap();
    let mut stream = Stream::new(&[9.0, 5.0, 3.0, 2.0, 1.0, 0.5], 5);
    for _ in 0..20_000 {
        m.step(&stream.row(), &[], 1.0, 1.0);
    }
    let (dirs, _) = m.directions();
    let q = m.cfg().kept();
    for i in 0..q {
        for j in 0..q {
            let g: f64 = (0..k).map(|t| dirs[i * k + t] * dirs[j * k + t]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((g - want).abs() <= 1e-14, "G[{i}][{j}] = {g}");
        }
    }
}

/// `ccipca` and `oja` are stochastic approximations: on a separated
/// spectrum each converges toward the covariance's eigenvectors, measured
/// against faer's eigensolver on the same rows. The bounds are statistical,
/// measured on this stream: after 20,000 rows without decay each direction
/// is within `1 - |cos| < 1e-3` of the truth (largest measured 2.3e-4,
/// `oja`'s second).
#[test]
fn the_stochastic_methods_converge_on_a_separated_spectrum() {
    let k = 5;
    for method in [PcaMethod::Ccipca, PcaMethod::Oja] {
        let mut m = StreamPca::new(cfg(k, 2, method, Decay::Halflife(f64::INFINITY))).unwrap();
        let mut stream = Stream::new(&[9.0, 4.0, 1.0, 0.5, 0.25], 21);
        let mut rows = Vec::new();
        for _ in 0..20_000 {
            let x = stream.row();
            m.step(&x, &[], 1.0, 1.0);
            rows.push((x, 1.0));
        }
        let (c, _) = covariance(&rows, 1.0);
        let (values, vectors) = crate::oracle::sym_eigen(&c);
        let (dirs, vals) = m.directions();
        let worst = worst_alignment(dirs, k, 2, &vectors);
        assert!(worst < 1e-3, "{}: 1 - |cos| {worst:e}", method.name());
        for j in 0..2 {
            let want = values[k - 1 - j];
            assert!(
                (vals[j] - want).abs() < 0.05 * want,
                "{}: component {j} variance {} against {want}",
                method.name(),
                vals[j]
            );
        }
        assert!(m.separated(0) && m.separated(1), "{}", method.name());
    }
}

/// The gate on both sides of its threshold: `(N - 1) ln(((l + m)/2)^2 /
/// (l m)) > 9.2103`. For `l = 4, m = 1` the log is `ln(25/16)`, so the
/// threshold falls between `N = 21` (20 ln(25/16) = 8.93) and `N = 22`
/// (9.37).
#[test]
fn the_eigengap_gate_holds_on_both_sides_of_its_threshold() {
    let tr = 10.0;
    let stat = |n: f64| (n - 1.0) * (25.0f64 / 16.0).ln();
    assert!(stat(21.0) < PCA_GAP_CRITICAL && stat(22.0) > PCA_GAP_CRITICAL);
    assert!(!separated(&[4.0, 1.0], 0, 21.0, tr, 2));
    assert!(!separated(&[4.0, 1.0], 1, 21.0, tr, 2));
    assert!(separated(&[4.0, 1.0], 0, 22.0, tr, 2));
    assert!(separated(&[4.0, 1.0], 1, 22.0, tr, 2));
    // A middle component needs both neighbors: apart from the one above
    // but not the one below.
    assert!(!separated(&[16.0, 4.0, 3.5], 1, 22.0, tr, 3));
    assert!(separated(&[16.0, 4.0, 1.0], 1, 22.0, tr, 3));
    // The last kept direction has no neighbor below.
    assert!(separated(&[16.0, 4.0], 1, 22.0, tr, 2));
    // A neighbor of no variance is told apart once the rows could have
    // given it some: at index 1, `N > 2`.
    assert!(!separated(&[4.0, 0.0], 0, 2.0, tr, 2));
    assert!(separated(&[4.0, 0.0], 0, 2.5, tr, 2));
    assert!(!separated(&[4.0, 2.0, 0.0], 1, 3.0, tr, 3));
    assert!(separated(&[400.0, 2.0, 0.0], 1, 3.5, tr, 3));
    // A variance at the trace's rounding is no direction; one just above
    // it is, and is a neighbor of some variance.
    let floor = tr * f64::EPSILON * 2.0;
    assert!(!separated(&[4.0, floor], 1, 1e6, tr, 2));
    assert!(separated(&[4.0, floor * 2.0], 1, 1e6, tr, 2));
    // No weight read yet, or one row.
    assert!(!separated(&[4.0, 1.0], 0, 0.0, tr, 2));
    assert!(!separated(&[4.0, 1.0], 0, 1.0, tr, 2));
}

/// No count of rows gates the first rows: they are withheld because Kish's
/// `N` is small, and the first component is emitted on the first row whose
/// state tells it apart. With unit weights and no decay `N` is the row
/// count, so on a well-separated stream that is a handful of rows, and on
/// an isotropic one (no gap at all) never.
#[test]
fn the_first_rows_are_withheld_by_the_gap_not_by_a_count() {
    let k = 3;
    let mut m = StreamPca::new(cfg(k, 1, PcaMethod::Svd, Decay::Halflife(f64::INFINITY))).unwrap();
    let mut stream = Stream::new(&[100.0, 1.0, 0.5], 3);
    let mut first = None;
    for i in 0..200 {
        let step = m.step(&stream.row(), &[], 1.0, 1.0);
        let shown = step.pred[0].is_finite();
        assert_eq!(shown, step.pred.iter().all(|v| v.is_finite()), "row {i}");
        if shown && first.is_none() {
            first = Some(i);
        }
    }
    let first = first.expect("a separated first component is emitted");
    // Measured: the fifth row (index 4), where `N = 4` and the test's
    // statistic on a ratio near 100 passes its critical value.
    assert!((2..10).contains(&first), "first emitted on row {first}");
    // An isotropic stream has no gap between its components.
    let mut m = StreamPca::new(cfg(k, 1, PcaMethod::Svd, Decay::Halflife(f64::INFINITY))).unwrap();
    let mut stream = Stream::new(&[1.0, 1.0, 1.0], 4);
    let mut shown = 0;
    for _ in 0..5000 {
        let step = m.step(&stream.row(), &[], 1.0, 1.0);
        shown += usize::from(step.pred[0].is_finite());
    }
    // A sample of an isotropic stream has gaps of its sampling error, which
    // the test at 1% reads as no gap on all but a few rows (measured: none
    // of the 5,000).
    assert!(shown < 50, "{shown} of 5000 rows told a tie apart");
}

/// A component emitted is its variance (unbiased, scikit-learn's `n - 1`
/// at unit weights), its share of the trace, its loadings and the row's
/// score on the mean and direction before the row.
#[test]
fn a_row_emits_the_components_before_it() {
    let k = 3;
    let mut m = StreamPca::new(cfg(k, 2, PcaMethod::Svd, Decay::Halflife(f64::INFINITY))).unwrap();
    let mut stream = Stream::new(&[25.0, 4.0, 0.25], 9);
    for _ in 0..500 {
        m.step(&stream.row(), &[], 1.0, 1.0);
    }
    let x = stream.row();
    let before = m.clone();
    let step = m.step(&x, &[], 1.0, 1.0);
    let (dirs, vals) = before.directions();
    let n = 500.0;
    for j in 0..2 {
        let slot = &step.pred[j * (k + 3)..(j + 1) * (k + 3)];
        assert_eq!(slot[0], n * n / (n * n - n) * vals[j]);
        assert_eq!(slot[1], vals[j] / before.trace());
        assert_eq!(&slot[2..2 + k], &dirs[j * k..(j + 1) * k]);
        let score: f64 = (0..k)
            .map(|i| dirs[j * k + i] * before.moments.deviation(i, x[i]))
            .fold(0.0, |s, v| s + v);
        assert_eq!(slot[2 + k], score);
    }
    assert_eq!(step.n_eff, n);
    assert_eq!(step.pred, before.predict(&x, 1.0).pred);
}

/// A row of weight 0 handed to the model decays the weight and moves
/// nothing else, to the bit, for every method; the next row decays as from
/// the row before it.
#[test]
fn a_zero_weight_row_moves_only_the_weight() {
    for method in [PcaMethod::Svd, PcaMethod::Ccipca, PcaMethod::Oja] {
        let mut m = StreamPca::new(cfg(4, 2, method, Decay::Lam(0.98))).unwrap();
        let mut stream = Stream::new(&[9.0, 4.0, 1.0, 0.5], 13);
        for _ in 0..50 {
            m.step(&stream.row(), &[], 1.0, 1.0);
        }
        let before = m.clone();
        m.step(&stream.row(), &[], 1.0, 0.0);
        assert_eq!(m.rows, before.rows, "{}", method.name());
        assert_eq!(m.vals, before.vals, "{}", method.name());
        assert_eq!(m.moments.means(), before.moments.means());
        assert_eq!(m.n_eff(), 0.98 * before.n_eff());
        assert!((m.n_kish() - before.n_kish()).abs() <= 1e-14 * before.n_kish());
    }
}

/// A first row of weight 0, and a stream that starts with several, leaves no
/// NaN anywhere: every division by the weight is guarded.
#[test]
fn zero_weight_first_rows_leave_no_nan() {
    for method in [PcaMethod::Svd, PcaMethod::Ccipca, PcaMethod::Oja] {
        let mut m = StreamPca::new(cfg(3, 2, method, Decay::Lam(0.98))).unwrap();
        let mut stream = Stream::new(&[16.0, 2.0, 1.0], 17);
        for _ in 0..5 {
            m.step(&stream.row(), &[], 1.0, 0.0);
        }
        for _ in 0..100 {
            m.step(&stream.row(), &[], 1.0, 1.0);
        }
        assert!(m.rows.iter().chain(&m.vals).all(|v| v.is_finite()));
        assert!(m.moments.means().iter().all(|v| v.is_finite()));
        assert!(m.separated(0), "{}", method.name());
    }
}

/// The sign rule: the first time a direction is seen, its largest-magnitude
/// entry is positive; after that it keeps the side it was on, so a score
/// series does not flip between two rows.
#[test]
fn signs_start_on_the_largest_entry_and_keep_their_side() {
    for method in [PcaMethod::Svd, PcaMethod::Ccipca, PcaMethod::Oja] {
        let k = 4;
        let mut m = StreamPca::new(cfg(k, 2, method, Decay::Lam(0.99))).unwrap();
        let mut stream = Stream::new(&[9.0, 4.0, 1.0, 0.5], 23);
        let mut prev = m.rows.clone();
        for i in 0..3000 {
            m.step(&stream.row(), &[], 1.0, 1.0);
            for j in 0..2 {
                let e = &m.rows[j * k..(j + 1) * k];
                let o = &prev[j * k..(j + 1) * k];
                if o.iter().all(|v| *v == 0.0) {
                    if e.iter().any(|v| *v != 0.0) {
                        let lead = e
                            .iter()
                            .enumerate()
                            .fold(0, |l, (t, v)| if v.abs() > e[l].abs() { t } else { l });
                        assert!(e[lead] > 0.0, "{} row {i}", method.name());
                    }
                } else {
                    let c: f64 = e.iter().zip(o).map(|(a, b)| a * b).sum();
                    assert!(c >= 0.0, "{} row {i}: component {j} flipped", method.name());
                }
            }
            prev = m.rows.clone();
        }
    }
}

/// A state saved mid-stream and restored continues to the bit, for every
/// method.
#[test]
fn a_saved_state_continues_to_the_bit() {
    for method in [PcaMethod::Svd, PcaMethod::Ccipca, PcaMethod::Oja] {
        let mut m = StreamPca::new(cfg(4, 2, method, Decay::Lam(0.97))).unwrap();
        let mut stream = Stream::new(&[9.0, 4.0, 1.0, 0.5], 29);
        for _ in 0..100 {
            m.step(&stream.row(), &[], 1.0, 1.0);
        }
        let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
        let state: State = rmp_serde::from_slice(&bytes).unwrap();
        let mut back = StreamPca::restore(&state).unwrap();
        assert_eq!(back, m);
        for _ in 0..100 {
            let x = stream.row();
            let a = m.step(&x, &[], 1.0, 1.0);
            let b = back.step(&x, &[], 1.0, 1.0);
            assert_eq!(
                a.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        assert_eq!(back, m);
    }
}

#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut m = StreamPca::new(cfg(3, 2, PcaMethod::Svd, Decay::Lam(0.9))).unwrap();
    m.vals.pop();
    let err = StreamPca::restore(&m.state()).unwrap_err().to_string();
    assert!(err.contains("wrong shape"), "{err}");
    let mut m = StreamPca::new(cfg(3, 2, PcaMethod::Svd, Decay::Lam(0.9))).unwrap();
    m.rows.push(0.0);
    assert!(StreamPca::restore(&m.state()).is_err());
}

/// The settings each refused with its remedy, and the state's size checked
/// against `max_state_bytes` at the call: `8 (2 + 3k + q k + q)` bytes, so
/// 2,000 features and 13 components (14 kept) hold 272,128 bytes, past the
/// default 256 KB, and pass with the cap raised.
#[test]
fn the_settings_are_checked_with_their_remedies() {
    let base = cfg(3, 2, PcaMethod::Svd, Decay::Lam(0.9));
    let refused = |c: StreamPcaCfg| StreamPca::new(c).unwrap_err();
    assert!(
        refused(StreamPcaCfg {
            n_components: 0,
            ..base.clone()
        })
        .contains("at least 1")
    );
    assert!(
        refused(StreamPcaCfg {
            n_components: 4,
            ..base.clone()
        })
        .contains("at most the number")
    );
    assert!(
        refused(StreamPcaCfg {
            learning_rate: Some(1.0),
            ..base.clone()
        })
        .contains("takes no rate")
    );
    let oja = StreamPcaCfg {
        method: PcaMethod::Oja,
        ..base.clone()
    };
    assert!(
        refused(StreamPcaCfg {
            learning_rate: Some(0.0),
            ..oja.clone()
        })
        .contains("above 0")
    );
    assert!(
        StreamPca::new(StreamPcaCfg {
            learning_rate: Some(0.5),
            ..oja
        })
        .is_ok()
    );
    assert!(
        refused(StreamPcaCfg {
            min_weight: -1.0,
            ..base.clone()
        })
        .contains("min_weight")
    );
    let wide = cfg(2000, 13, PcaMethod::Svd, Decay::Lam(0.9));
    assert_eq!(wide.state_bytes(), 8 * (2 + 6000 + 14 * 2000 + 14));
    assert!(wide.state_bytes() > PCA_MAX_STATE_BYTES);
    assert!(refused(wide.clone()).contains("past max_state_bytes of 262144"));
    assert!(
        StreamPca::new(StreamPcaCfg {
            max_state_bytes: Some(300_000),
            ..wide
        })
        .is_ok()
    );
    // One kept beside the emitted, but never more than the features.
    assert_eq!(cfg(3, 3, PcaMethod::Svd, Decay::Lam(0.9)).kept(), 3);
    assert_eq!(cfg(3, 2, PcaMethod::Svd, Decay::Lam(0.9)).kept(), 3);
    assert_eq!(cfg(5, 2, PcaMethod::Svd, Decay::Lam(0.9)).kept(), 3);
}

/// The state's size is the shape's, whatever the stream's length: the
/// encoded state after 10 rows and after 10,000 is the same number of
/// bytes.
#[test]
fn the_state_does_not_grow_with_the_stream() {
    for method in [PcaMethod::Svd, PcaMethod::Ccipca, PcaMethod::Oja] {
        let mut m = StreamPca::new(cfg(5, 2, method, Decay::Lam(0.99))).unwrap();
        let mut stream = Stream::new(&[9.0, 4.0, 1.0, 0.5, 0.2], 31);
        for _ in 0..10 {
            m.step(&stream.row(), &[], 1.0, 1.0);
        }
        let early = rmp_serde::to_vec_named(&m.state()).unwrap().len();
        for _ in 0..10_000 {
            m.step(&stream.row(), &[], 1.0, 1.0);
        }
        let late = rmp_serde::to_vec_named(&m.state()).unwrap().len();
        assert_eq!(early, late, "{}", method.name());
    }
}

/// Under decay the components follow a change: a stream that rotates its
/// basis half way is read on the new basis a few half-lives later.
#[test]
fn under_decay_the_components_follow_a_change() {
    let k = 4;
    for method in [PcaMethod::Svd, PcaMethod::Ccipca, PcaMethod::Oja] {
        let mut m = StreamPca::new(cfg(k, 1, method, Decay::Halflife(200.0))).unwrap();
        let mut a = Stream::new(&[9.0, 2.0, 1.0, 0.5], 37);
        let mut b = Stream::new(&[9.0, 2.0, 1.0, 0.5], 41);
        for _ in 0..3000 {
            m.step(&a.row(), &[], 1.0, 1.0);
        }
        for _ in 0..3000 {
            m.step(&b.row(), &[], 1.0, 1.0);
        }
        let c: f64 = m.rows[..k]
            .iter()
            .zip(&b.basis[0])
            .map(|(x, y)| x * y)
            .sum();
        assert!(
            1.0 - c.abs() < 0.01,
            "{}: 1 - |cos| {}",
            method.name(),
            1.0 - c.abs()
        );
    }
}

/// The rotation on rows already orthogonal moves nothing; on two rows it
/// leaves them orthogonal with the same total length.
#[test]
fn the_rotation_orthogonalizes_and_keeps_the_length() {
    let mut s = vec![3.0, 0.0, 0.0, 2.0];
    orthogonalize_rows(&mut s, 2, 2);
    assert_eq!(s, vec![3.0, 0.0, 0.0, 2.0]);
    let mut s = vec![1.0, 2.0, 3.0, 1.0, 0.5, -1.0, 2.0, 2.0, 1.0];
    let before: f64 = s.iter().map(|v| v * v).sum();
    orthogonalize_rows(&mut s, 3, 3);
    let after: f64 = s.iter().map(|v| v * v).sum();
    assert!((before - after).abs() <= 1e-14 * before);
    for p in 0..3 {
        for r in (p + 1)..3 {
            let g = dot(row(&s, p, 3), row(&s, r, 3));
            assert!(g.abs() <= 1e-14 * before, "{g}");
        }
    }
}

/// `predict` is the step without the step, on a stream whose components
/// differ (the shared contract's stream is isotropic, which `pca` withholds
/// by design): unequal weights, zero weights and clock gaps, every method,
/// the same bits row by row, and most rows emitting every slot.
#[test]
fn predict_is_the_step_on_a_separated_stream() {
    for method in [PcaMethod::Svd, PcaMethod::Ccipca, PcaMethod::Oja] {
        // `oja` at a gain its rate finds the second gap with (8 · 3.75 /
        // 29.25 = 1.03; the default's 0.51 sits on the condition's edge)
        let mut c = cfg(3, 2, method, Decay::Halflife(20.0));
        if method == PcaMethod::Oja {
            c.learning_rate = Some(8.0);
        }
        let mut m = StreamPca::new(c).unwrap();
        let mut stream = Stream::new(&[25.0, 4.0, 0.25], 43);
        let mut s = 7u64;
        let mut ready = 0;
        for i in 0..400 {
            let x = stream.row();
            let u = uniform(&mut s);
            let w = if u < 0.05 { 0.0 } else { 0.5 + u };
            let d = if i == 0 {
                0.0
            } else if u > 0.95 {
                25.0
            } else {
                1.0
            };
            let p = m.predict(&x, d);
            let step = m.step(&x, &[], d, w);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&p.pred), bits(&step.pred), "{} row {i}", method.name());
            assert_eq!(p.n_eff.to_bits(), step.n_eff.to_bits());
            ready += usize::from(step.pred.iter().all(|v| v.is_finite()));
        }
        // Measured: 341 for `svd`, 349 for `ccipca`, 205 for `oja`, whose
        // directions are the noisiest at a half-life of 20 rows (79 at the
        // default gain, where its rate's condition on the second gap
        // fails on most rows; 272 before that condition was read).
        let least = if method == PcaMethod::Oja { 180 } else { 300 };
        assert!(ready > least, "{}: {ready} rows ready", method.name());
    }
}

/// The share of the second half of `n` rows that writes component 0, on
/// a stream of `spectrum` (k = its length, one component), and `oja`'s
/// gain `rate` where given.
fn written_late(
    method: PcaMethod,
    spectrum: &[f64],
    seed: u64,
    n: usize,
    rate: Option<f64>,
) -> f64 {
    let mut c = cfg(spectrum.len(), 1, method, Decay::Halflife(f64::INFINITY));
    c.learning_rate = rate;
    let mut m = StreamPca::new(c).unwrap();
    let mut stream = Stream::new(spectrum, seed);
    let mut shown = 0;
    for i in 0..n {
        let step = m.step(&stream.row(), &[], 1.0, 1.0);
        if i >= n / 2 {
            shown += usize::from(step.pred[0].is_finite());
        }
    }
    shown as f64 / (n - n / 2) as f64
}

/// A tie at the top of a wide spectrum is withheld by every method, not
/// only `svd` (review W13, C1): k = 20, variances (4, 4, 1, ..., 1), one
/// component, which any direction of the tied plane is. `oja`'s variances
/// are running means along directions still moving, and at the default gain
/// its second direction does not converge -- a rate `c/t` finds the
/// direction below a gap only where `c · gap > 1/2`, and here `c = 4/26`
/// per unit of variance against a gap of 3, 0.46 -- so they are not
/// eigenvalue estimates, and the eigengap test read their convergence
/// error as a gap: on 3 of 8 seeds it wrote the unidentified component on
/// every row of the second half of 20,000. Gated on that condition, it
/// writes none; `svd` and `ccipca`, whose variances are eigenvalue
/// estimates, write none late either. An isotropic stream is withheld by
/// all three. At twice the gain the rate finds the tied plane, and the
/// test reads the tie. The gate withholds only what has not converged: a
/// gap the rate finds (16 against 1, `c · gap = 4 · 15/35`) is written on
/// every row of the second half.
#[test]
fn a_tie_is_withheld_by_every_method() {
    let mut tie = vec![1.0; 20];
    tie[0] = 4.0;
    tie[1] = 4.0;
    let flat = vec![1.0; 20];
    let mut gap = vec![1.0; 20];
    gap[0] = 16.0;
    for seed in 0..8u64 {
        for method in [PcaMethod::Svd, PcaMethod::Ccipca, PcaMethod::Oja] {
            let late = written_late(method, &tie, 100 + seed, 20_000, None);
            assert!(late < 0.02, "{} tie, seed {seed}: {late}", method.name());
            let late = written_late(method, &flat, 200 + seed, 10_000, None);
            assert!(late < 0.02, "{} flat, seed {seed}: {late}", method.name());
        }
        let late = written_late(PcaMethod::Oja, &tie, 100 + seed, 20_000, Some(8.0));
        assert!(late < 0.02, "oja at a gain of 8, seed {seed}: {late}");
        let late = written_late(PcaMethod::Oja, &gap, 300 + seed, 10_000, None);
        assert_eq!(late, 1.0, "oja on a gap, seed {seed}");
    }
}

/// `oja`'s rate condition on both sides of 1/2: `learning_rate · gap / tr`
/// against each neighbor, the one above and the one below.
#[test]
fn the_rate_condition_holds_on_both_sides_of_one_half() {
    // gap 3, trace 26: the default gain of 4 gives 0.4615, a gain of 4.4
    // gives 0.5077
    let vals = [4.0, 1.0];
    assert!(!rate_finds(&vals, 0, 26.0, 4.0));
    assert!(!rate_finds(&vals, 1, 26.0, 4.0));
    assert!(rate_finds(&vals, 0, 26.0, 4.4));
    assert!(rate_finds(&vals, 1, 26.0, 4.4));
    // exactly 1/2 is not past it
    assert!(!rate_finds(&vals, 0, 24.0, 4.0));
    // a middle component needs both: found against the one above, not the
    // one below
    assert!(!rate_finds(&[16.0, 4.0, 3.5], 1, 20.0, 4.0));
    assert!(rate_finds(&[16.0, 4.0, 1.0], 1, 20.0, 4.0));
    // the gate reads it for oja alone
    let mut m = StreamPca::new(cfg(2, 1, PcaMethod::Svd, Decay::Halflife(f64::INFINITY))).unwrap();
    m.vals = vec![4.0, 1.0];
    assert!(m.rate_finds(0, 26.0));
    m.cfg.method = PcaMethod::Oja;
    assert!(!m.rate_finds(0, 26.0));
}

/// The rank rule reads Kish's effective count, which under a decay stops
/// near `(1 + lam)/(1 - lam)` (Fable's W13 review, C2): on data of exact
/// rank 2 the second of two components, whose neighbor has no variance,
/// needs that count past 3, so at a half-life of 0.9 rows (2.7) it is never
/// written, and at 3 rows (8.7) it is. The rows of positive weight would
/// read a row whose weight has decayed to nothing as one that spans the
/// data (`model_contract`'s zero-weight row past the underflow), so the
/// limit is the docs', on both sides here.
#[test]
fn the_rank_rule_under_a_half_life_of_a_row_is_kishs_count() {
    let k = 3;
    for (half_life, writes) in [(0.9, false), (3.0, true)] {
        let mut m = StreamPca::new(cfg(k, 2, PcaMethod::Svd, Decay::Halflife(half_life))).unwrap();
        let mut stream = Stream::new(&[100.0, 0.001, 0.0], 21);
        let mut written = 0;
        for _ in 0..200 {
            m.step(&stream.row(), &[], 1.0, 1.0);
            written += usize::from(m.separated(1));
        }
        let (_, vals) = m.directions();
        assert!(vals[2] <= 1e-12 * vals[0], "{vals:?}");
        let lam = 0.5f64.powf(1.0 / half_life);
        let cap = (1.0 + lam) / (1.0 - lam);
        assert!(
            (m.n_kish() - cap).abs() < 1e-9 * cap,
            "{} {cap}",
            m.n_kish()
        );
        if writes {
            assert!(
                written > 150,
                "{half_life}: written on {written} of 200 rows"
            );
        } else {
            assert_eq!(written, 0, "{half_life}");
        }
    }
}
