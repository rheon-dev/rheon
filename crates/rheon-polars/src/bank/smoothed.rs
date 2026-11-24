//! `kalman`'s fixed-lag smoother read from the state: the observations the
//! stream has not yet completed the lag of, each group's last `L` a target,
//! smoothed with every row learned so far. The output carries a value on
//! the row that completes a lag, and the stream has no row after its last
//! to carry these, so they are read here instead.

use polars::prelude::*;
use rheon_core::ClockValue;

use super::{Bank, GroupKey, closed_rows::instance_decay};
use crate::spec::ModelKind;
use crate::stream::{AnyModel, ClockDtype};

impl Bank {
    /// For a `kalman` spec with `smooth_lag`, one row per group, decay
    /// instance, target and pending observation, oldest first: `spec`,
    /// `group`, `instance`, `half_life`, `lam`, `target`, `smoothed_for`
    /// (the observation's clock, in the clock column's type, or its row
    /// number with no clock column), `withheld` (`"below_min_weight"`
    /// where the gate withheld that row's prediction, `"not_observed"` for a
    /// target its shared covariance had not yet seen, null where the value
    /// stands), `smoothed_coef` (laid out like one target's `coef`, null
    /// where nothing is estimated) and, under `emit_smoothed_se`,
    /// `smoothed_se_coef`. The newest observation's value is its filtered
    /// one; each older one is smoothed with the rows after it.
    ///
    /// `group` narrows the frame to those keys, in key order; a key the
    /// bank has never seen gives nothing, not an error.
    ///
    /// # Errors
    ///
    /// `spec` out of range, or not a `kalman` spec with `smooth_lag`.
    pub fn smoothed_tail(
        &self,
        spec: usize,
        group: Option<&[GroupKey]>,
    ) -> Result<DataFrame, String> {
        let keys = self.sorted_keys(spec, group)?;
        let (s, states) = (&self.specs[spec], &self.states[spec]);
        if !matches!(s.model, ModelKind::Kalman { .. }) || s.smooth_lag().is_none() {
            return Err(format!(
                "spec {:?}: it has no fixed-lag smoother, which only a kalman spec with \
                 smooth_lag keeps. Name a kalman spec that sets smooth_lag.",
                s.name
            ));
        }
        let se = s.emit_smoothed_se();
        let (mut group_col, mut instance, mut target) = (vec![], vec![], vec![]);
        let (mut label, mut withheld) = (Vec::<Option<ClockValue>>::new(), vec![]);
        let (mut coef, mut ses): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (vec![], vec![]);
        for key in keys {
            for (inst, model) in &states[key].models {
                let AnyModel::Kalman(m) = model else {
                    unreachable!("a kalman spec builds kalman models");
                };
                for row in m.smoothed_tail().unwrap_or_default() {
                    group_col.push(key.as_str());
                    instance.push(inst.as_str());
                    target.push(s.targets[row.target].as_str());
                    label.push(Some(row.label));
                    withheld.push(match row.held {
                        rheon_core::SMOOTH_STOOD => None,
                        rheon_core::SMOOTH_MIN_WEIGHT => Some("below_min_weight"),
                        _ => Some("not_observed"),
                    });
                    ses.push(row.var.iter().map(|v| v.max(0.0).sqrt()).collect());
                    coef.push(row.coef);
                }
            }
        }
        let n = coef.len();
        let decay: Vec<(Option<f64>, Option<f64>)> =
            instance.iter().map(|i| instance_decay(s, i)).collect();
        // As the output's `smoothed_for` is: with no clock column, the row's
        // index in its group, an integer.
        let numeric = ClockDtype::Numeric;
        let clock = self.clock_dtypes[spec].as_ref().unwrap_or(&numeric);
        let labels = Series::from_arrow("smoothed_for".into(), clock.array(&label))
            .map_err(|e| e.to_string())?;
        let lists = |name: &str, v: &[Vec<f64>]| -> Column {
            let mut b = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                name.into(),
                v.len(),
                v.first().map_or(0, Vec::len),
                DataType::Float64,
            );
            for l in v {
                b.append_iter(l.iter().map(|x| (!x.is_nan()).then_some(*x)));
            }
            b.finish().into_column()
        };
        let mut cols = vec![
            Column::new("spec".into(), vec![s.name.as_str(); n]),
            Column::new("group".into(), group_col),
            Column::new("instance".into(), instance),
            Column::new(
                "half_life".into(),
                decay.iter().map(|d| d.0).collect::<Vec<_>>(),
            ),
            Column::new("lam".into(), decay.iter().map(|d| d.1).collect::<Vec<_>>()),
            Column::new("target".into(), target),
            labels.into_column(),
            Column::new("withheld".into(), withheld),
            lists("smoothed_coef", &coef),
        ];
        if se {
            cols.push(lists("smoothed_se_coef", &ses));
        }
        DataFrame::new(n, cols).map_err(|e| e.to_string())
    }
}
