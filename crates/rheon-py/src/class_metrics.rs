//! PyO3 binding for streaming binary classification metrics (`ClassMetrics` and
//! `ReliabilityBins` from `rheon-core`).

use polars::prelude::{DataType, Series};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PySeries;
use rheon_polars::rheon_core::{
    ClassMetrics, DEFAULT_CLASS_METRICS_BINS, DEFAULT_CLASS_METRICS_COMPRESSION, ReliabilityBins,
};

enum Inner {
    Full(Box<ClassMetrics>),
    BinsOnly(ReliabilityBins),
}

type SummaryTuple = (
    u64,
    u64,
    u64,
    f64,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
);

type BinRowTuple = (
    usize,
    f64,
    f64,
    u64,
    f64,
    Option<f64>,
    Option<f64>,
    Option<f64>,
);

fn to_f64_series(s: &Series, name: &str) -> PyResult<Series> {
    if s.dtype() == &DataType::Float64 {
        return Ok(s.clone());
    }
    if s.dtype() == &DataType::Boolean {
        let u = s.cast(&DataType::UInt8).map_err(|e| {
            PyValueError::new_err(format!(
                "ClassMetrics: {name} could not be cast from boolean ({e}). Cast {name} to Float64."
            ))
        })?;
        return u.cast(&DataType::Float64).map_err(|e| {
            PyValueError::new_err(format!(
                "ClassMetrics: {name} could not be cast to float64 ({e}). Cast {name} to Float64."
            ))
        });
    }
    s.cast(&DataType::Float64).map_err(|e| {
        PyValueError::new_err(format!(
            "ClassMetrics: {name} could not be cast to float64 ({e}). Cast {name} to Float64."
        ))
    })
}

/// Streaming binary classification metrics accumulator backed by two merging
/// per-class t-digests and exact uniform reliability bins.
#[pyclass(name = "ClassMetrics", module = "rheon._rheon")]
pub struct PyClassMetrics {
    inner: Inner,
}

#[pymethods]
impl PyClassMetrics {
    #[new]
    #[pyo3(signature = (
        bins = DEFAULT_CLASS_METRICS_BINS,
        compression = DEFAULT_CLASS_METRICS_COMPRESSION,
        max_state_bytes = None,
        bins_only = false,
    ))]
    fn new(
        bins: usize,
        compression: f64,
        max_state_bytes: Option<u64>,
        bins_only: bool,
    ) -> PyResult<Self> {
        let inner = if bins_only {
            Inner::BinsOnly(
                ReliabilityBins::new(bins, max_state_bytes).map_err(PyValueError::new_err)?,
            )
        } else {
            Inner::Full(Box::new(
                ClassMetrics::new(bins, compression, max_state_bytes)
                    .map_err(PyValueError::new_err)?,
            ))
        };
        Ok(Self { inner })
    }

    /// Fold one chunk of `(preds, ys, weights)` Polars Series in arrival order.
    #[pyo3(signature = (preds, ys, weights = None))]
    fn update_batch(
        &mut self,
        preds: PySeries,
        ys: PySeries,
        weights: Option<PySeries>,
    ) -> PyResult<()> {
        let n = preds.0.len();
        if ys.0.len() != n {
            return Err(PyValueError::new_err(format!(
                "ClassMetrics: preds length ({n}) does not match ys length ({}). \
                 Pass Series of matching length.",
                ys.0.len()
            )));
        }
        if let Some(ref ws) = weights
            && ws.0.len() != n
        {
            return Err(PyValueError::new_err(format!(
                "ClassMetrics: weights length ({}) does not match preds length ({n}). \
                 Pass Series of matching length.",
                ws.0.len()
            )));
        }
        if n == 0 {
            return Ok(());
        }

        let ps = to_f64_series(&preds.0, "preds")?;
        let ys_f = to_f64_series(&ys.0, "ys")?;
        let ws_f = match weights {
            Some(ref w) => Some(to_f64_series(&w.0, "weights")?),
            None => None,
        };

        let p_ca = ps
            .f64()
            .map_err(|e| PyValueError::new_err(format!("preds: {e}")))?;
        let y_ca = ys_f
            .f64()
            .map_err(|e| PyValueError::new_err(format!("ys: {e}")))?;

        match (&mut self.inner, ws_f) {
            (Inner::Full(cm), None) => {
                for (po, yo) in p_ca.iter().zip(y_ca.iter()) {
                    if let (Some(p), Some(y)) = (po, yo) {
                        cm.update(p, y, 1.0).map_err(PyValueError::new_err)?;
                    }
                }
            }
            (Inner::Full(cm), Some(ws)) => {
                let w_ca = ws
                    .f64()
                    .map_err(|e| PyValueError::new_err(format!("weights: {e}")))?;
                for ((po, yo), wo) in p_ca.iter().zip(y_ca.iter()).zip(w_ca.iter()) {
                    if let (Some(p), Some(y), Some(w)) = (po, yo, wo) {
                        cm.update(p, y, w).map_err(PyValueError::new_err)?;
                    }
                }
            }
            (Inner::BinsOnly(bins), None) => {
                for (po, yo) in p_ca.iter().zip(y_ca.iter()) {
                    if let (Some(p), Some(y)) = (po, yo) {
                        bins.update(p, y, 1.0).map_err(PyValueError::new_err)?;
                    }
                }
            }
            (Inner::BinsOnly(bins), Some(ws)) => {
                let w_ca = ws
                    .f64()
                    .map_err(|e| PyValueError::new_err(format!("weights: {e}")))?;
                for ((po, yo), wo) in p_ca.iter().zip(y_ca.iter()).zip(w_ca.iter()) {
                    if let (Some(p), Some(y), Some(w)) = (po, yo, wo) {
                        bins.update(p, y, w).map_err(PyValueError::new_err)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Number of positive-weight usable observations folded.
    #[getter]
    fn n(&self) -> u64 {
        match &self.inner {
            Inner::Full(cm) => cm.n(),
            Inner::BinsOnly(bins) => bins.n(),
        }
    }

    /// Merge another compatible `ClassMetrics` state into `self`.
    fn merge(&mut self, other: &Self) -> PyResult<()> {
        match (&mut self.inner, &other.inner) {
            (Inner::Full(a), Inner::Full(b)) => a.merge(b).map_err(PyValueError::new_err),
            (Inner::BinsOnly(a), Inner::BinsOnly(b)) => a.merge(b).map_err(PyValueError::new_err),
            _ => Err(PyValueError::new_err(
                "ClassMetrics: cannot merge a full state with a bins-only state. \
                 Pass two states built with the same bins_only setting.",
            )),
        }
    }

    /// Summary tuple `(n, n_pos, n_neg, weight_sum, pos_rate, roc_auc, pr_auc, ece, mce, brier)`.
    fn summary(&self) -> PyResult<SummaryTuple> {
        match &self.inner {
            Inner::Full(cm) => {
                let s = cm.summary();
                Ok((
                    s.n,
                    s.n_pos,
                    s.n_neg,
                    s.weight_sum,
                    s.pos_rate,
                    s.roc_auc,
                    s.pr_auc,
                    s.ece,
                    s.mce,
                    s.brier,
                ))
            }
            Inner::BinsOnly(_) => Err(PyValueError::new_err(
                "ClassMetrics: summary is unavailable when bins_only=True. \
                 Set bins_only=False to read summary().",
            )),
        }
    }

    /// Exact reliability bin rows `(bin, bin_lo, bin_hi, n, weight_sum, mean_pred, mean_actual, abs_gap)`.
    fn bins(&self) -> Vec<BinRowTuple> {
        let rows = match &self.inner {
            Inner::Full(cm) => cm.reliability_bins(),
            Inner::BinsOnly(bins) => bins.rows(),
        };
        rows.into_iter()
            .map(|r| {
                (
                    r.bin,
                    r.bin_lo,
                    r.bin_hi,
                    r.n,
                    r.weight_sum,
                    r.mean_pred,
                    r.mean_actual,
                    r.abs_gap,
                )
            })
            .collect()
    }
}
