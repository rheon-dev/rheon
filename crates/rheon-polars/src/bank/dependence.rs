//! `Bank::dependence`: tidy per-`(feature, target)` table of current nonlinear
//! dependence readings (MI, Miller-Madow MI, Spearman's rho, Kendall's tau-b,
//! Chatterjee's xi) for every group and decay instance of a `dependence` spec.

use super::*;

impl Bank {
    /// Current nonlinear dependence measures for every `(feature, target)` pair
    /// of a `dependence` spec, across every group and decay instance.
    ///
    /// Returns a tidy Polars `DataFrame` with columns `spec`, `group`,
    /// `instance`, `half_life`, `lam`, `feature`, `target`, `weight_sum`, and
    /// one column per active measure in `measures` order (`mi`, `mi_mm`,
    /// `spearman`, `kendall`, `xi`). Periodic rank measures (`spearman`, `xi`)
    /// are computed fresh from the current window so the table always reflects
    /// the latest rows learned, even when `rank_every` or `rank_every_rows`
    /// throttles per-row output.
    ///
    /// Respects `min_settled_frac` and per-target `min_weight`: measures are
    /// reported as null while a target or stream is below its readiness floor.
    ///
    /// # Errors
    ///
    /// `spec` out of range, or not a `dependence` spec.
    pub fn dependence(&self, spec: usize, group: Option<&[GroupKey]>) -> Result<DataFrame, String> {
        let keys = self.sorted_keys(spec, group)?;
        let (s, states) = (&self.specs[spec], &self.states[spec]);
        let ModelKind::Dependence(d) = &s.model else {
            return Err(format!(
                "spec {:?}: its model is {:?}, not \"dependence\", and only a dependence spec \
                 keeps nonlinear dependence tables. Name a dependence spec, or read a marginal \
                 spec's pairs with ModelBank.marginal.",
                s.name,
                s.model.kind_name()
            ));
        };
        let measures = d.measures_for_fields();
        let min_weights = s.min_periods_per_target();
        let k = s.features.len();

        let mut group_col: Vec<Option<&str>> = Vec::new();
        let mut instance: Vec<&str> = Vec::new();
        let mut feature: Vec<&str> = Vec::new();
        let mut target: Vec<&str> = Vec::new();
        let mut pairs: Vec<rheon_core::DependencePair> = Vec::new();

        for key in keys {
            let stream = &states[key];
            for (i, (label, model)) in stream.models.iter().enumerate() {
                let AnyModel::Dependence(m) = model else {
                    unreachable!("a dependence spec builds dependence models");
                };
                let unsettled = stream.settled_withholds(s, i);
                let fresh = m.read_pairs_now();
                for (t_i, t) in s.targets.iter().enumerate() {
                    let min_w = min_weights.get(t_i).copied().unwrap_or(0.0);
                    for (j, f) in s.features.iter().enumerate() {
                        group_col.push(key.as_str());
                        instance.push(label.as_str());
                        feature.push(f.as_str());
                        target.push(t.as_str());
                        let mut p = fresh[t_i * k + j];
                        if unsettled || p.n_eff < min_w {
                            p.mi = f64::NAN;
                            p.mi_mm = f64::NAN;
                            p.spearman = f64::NAN;
                            p.kendall = f64::NAN;
                            p.xi = f64::NAN;
                        }
                        pairs.push(p);
                    }
                }
            }
        }

        let num = |f: fn(&rheon_core::DependencePair) -> f64| -> Vec<Option<f64>> {
            pairs
                .iter()
                .map(f)
                .map(|v| (!v.is_nan()).then_some(v))
                .collect()
        };
        let decay: Vec<(Option<f64>, Option<f64>)> =
            instance.iter().map(|i| instance_decay(s, i)).collect();
        let mut cols = vec![
            Column::new("spec".into(), vec![s.name.as_str(); pairs.len()]),
            Column::new("group".into(), group_col),
            Column::new("instance".into(), instance),
            Column::new(
                "half_life".into(),
                decay.iter().map(|d| d.0).collect::<Vec<_>>(),
            ),
            Column::new("lam".into(), decay.iter().map(|d| d.1).collect::<Vec<_>>()),
            Column::new("feature".into(), feature),
            Column::new("target".into(), target),
            Column::new("weight_sum".into(), num(|p| p.n_eff)),
        ];
        for m in measures {
            let col = match m {
                rheon_core::DependenceMeasure::Mi => Column::new("mi".into(), num(|p| p.mi)),
                rheon_core::DependenceMeasure::MiMm => {
                    Column::new("mi_mm".into(), num(|p| p.mi_mm))
                }
                rheon_core::DependenceMeasure::Spearman => {
                    Column::new("spearman".into(), num(|p| p.spearman))
                }
                rheon_core::DependenceMeasure::Kendall => {
                    Column::new("kendall".into(), num(|p| p.kendall))
                }
                rheon_core::DependenceMeasure::Xi => Column::new("xi".into(), num(|p| p.xi)),
            };
            cols.push(col);
        }
        DataFrame::new(pairs.len(), cols).map_err(|e| e.to_string())
    }
}
