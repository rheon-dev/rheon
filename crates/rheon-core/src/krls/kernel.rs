//! Positive-definite Mercer kernels for [`super::Krls`].
//!
//! Every exponential calls [`libm::exp`] rather than the platform math library
//! so kernel evaluations are identical to the bit across macOS, Linux and
//! Windows. Polynomial powers use integer exponentiation by squaring (pure
//! multiplications, no libm call).

use serde::{Deserialize, Serialize};

/// The default degree of [`KernelKind::Poly`], matching scikit-learn's
/// `KernelRidge(kernel="poly", degree=3)`.
pub const DEFAULT_POLY_DEGREE: u32 = 3;

/// The default offset of [`KernelKind::Poly`], matching scikit-learn's
/// `KernelRidge(kernel="poly", coef0=1.0)`.
pub const DEFAULT_POLY_COEF0: f64 = 1.0;

/// Positive-definite kernel family for [`super::Krls`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelKind {
    /// Radial basis function: `exp(-gamma * ||x - y||_2^2)`.
    #[default]
    Rbf,
    /// Laplacian kernel: `exp(-gamma * ||x - y||_1)`.
    Laplacian,
    /// Polynomial kernel: `(gamma * x^T y + coef0)^degree`.
    Poly,
    /// Linear kernel: `x^T y`.
    Linear,
}

impl KernelKind {
    /// The canonical spec word for this kernel.
    pub fn as_str(self) -> &'static str {
        match self {
            KernelKind::Rbf => "rbf",
            KernelKind::Laplacian => "laplacian",
            KernelKind::Poly => "poly",
            KernelKind::Linear => "linear",
        }
    }
}

/// Resolved kernel parameters ready for per-row evaluation.
#[derive(Debug, Clone, Copy)]
pub(super) struct KernelEval {
    pub kind: KernelKind,
    pub gamma: f64,
    pub degree: u32,
    pub coef0: f64,
}

impl KernelEval {
    /// Evaluate `k(x, y)`.
    #[inline]
    pub fn eval(self, x: &[f64], y: &[f64]) -> f64 {
        match self.kind {
            KernelKind::Rbf => {
                let mut d2 = 0.0;
                for (&xi, &yi) in x.iter().zip(y) {
                    let d = xi - yi;
                    d2 += d * d;
                }
                libm::exp(-self.gamma * d2)
            }
            KernelKind::Laplacian => {
                let mut d1 = 0.0;
                for (&xi, &yi) in x.iter().zip(y) {
                    d1 += (xi - yi).abs();
                }
                libm::exp(-self.gamma * d1)
            }
            KernelKind::Poly => {
                let mut dot = 0.0;
                for (&xi, &yi) in x.iter().zip(y) {
                    dot += xi * yi;
                }
                pow_u32(self.gamma * dot + self.coef0, self.degree)
            }
            KernelKind::Linear => {
                let mut dot = 0.0;
                for (&xi, &yi) in x.iter().zip(y) {
                    dot += xi * yi;
                }
                dot
            }
        }
    }

    /// Evaluate `k(x, x)`. Exact `1.0` for shift-invariant kernels (`Rbf`,
    /// `Laplacian`).
    #[inline]
    pub fn eval_self(self, x: &[f64]) -> f64 {
        match self.kind {
            KernelKind::Rbf | KernelKind::Laplacian => 1.0,
            KernelKind::Poly => {
                let mut n2 = 0.0;
                for &xi in x {
                    n2 += xi * xi;
                }
                pow_u32(self.gamma * n2 + self.coef0, self.degree)
            }
            KernelKind::Linear => {
                let mut n2 = 0.0;
                for &xi in x {
                    n2 += xi * xi;
                }
                n2
            }
        }
    }

    /// Fill `out` with `k(d_i, x)` for each atom `i` in a contiguous row-major
    /// matrix `atoms` of shape `m x k`.
    pub fn eval_slice(self, atoms: &[f64], m: usize, k: usize, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.reserve(m);
        for i in 0..m {
            let row = &atoms[i * k..(i + 1) * k];
            out.push(self.eval(row, x));
        }
    }

    /// Fill `out` with `k(d_i, x)` for each row `i` in a `VecDeque` storing `m`
    /// rows of width `k` (window mode).
    pub fn eval_deque(
        self,
        atoms: &std::collections::VecDeque<f64>,
        m: usize,
        k: usize,
        x: &[f64],
        out: &mut Vec<f64>,
    ) {
        out.clear();
        out.reserve(m);
        let (front, back) = atoms.as_slices();
        let front_len = front.len();
        for i in 0..m {
            let start = i * k;
            let end = start + k;
            let val = if end <= front_len {
                self.eval(&front[start..end], x)
            } else if start >= front_len {
                self.eval(&back[start - front_len..end - front_len], x)
            } else {
                self.eval_split(&front[start..front_len], &back[..end - front_len], x)
            };
            out.push(val);
        }
    }

    /// Evaluate `k(row, x)` when `row` wraps across the ring buffer's split.
    #[cold]
    fn eval_split(self, a: &[f64], b: &[f64], x: &[f64]) -> f64 {
        let (xa, xb) = x.split_at(a.len());
        match self.kind {
            KernelKind::Rbf => {
                let mut d2 = 0.0;
                for (&ri, &xi) in a.iter().zip(xa).chain(b.iter().zip(xb)) {
                    let d = ri - xi;
                    d2 += d * d;
                }
                libm::exp(-self.gamma * d2)
            }
            KernelKind::Laplacian => {
                let mut d1 = 0.0;
                for (&ri, &xi) in a.iter().zip(xa).chain(b.iter().zip(xb)) {
                    d1 += (ri - xi).abs();
                }
                libm::exp(-self.gamma * d1)
            }
            KernelKind::Poly => {
                let mut dot = 0.0;
                for (&ri, &xi) in a.iter().zip(xa).chain(b.iter().zip(xb)) {
                    dot += ri * xi;
                }
                pow_u32(self.gamma * dot + self.coef0, self.degree)
            }
            KernelKind::Linear => {
                let mut dot = 0.0;
                for (&ri, &xi) in a.iter().zip(xa).chain(b.iter().zip(xb)) {
                    dot += ri * xi;
                }
                dot
            }
        }
    }
}

/// Deterministic integer power by binary exponentiation (no libm call,
/// bit-identical across OSes).
#[inline]
pub(super) fn pow_u32(mut base: f64, mut exp: u32) -> f64 {
    let mut acc = 1.0;
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base;
        }
    }
    acc
}
