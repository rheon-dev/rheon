//! A `dist_fit` as the bank steps it: the model's slots per target, then
//! one more, the reason the target reports nothing on the row
//! ([`rheon_core::DistFit::withheld`]), read before the row is learned. The
//! bank lays a target's slots out together, so the reason sits last in each
//! target's group, where the per-target `min_weight` gate reaches it with
//! the rest.

use rheon_core::{DistFit, DistFitCfg, DistWithheld, OnlineModel, State, StateError, Step};

/// The reasons a `withheld_reason_<t>` field takes, in the order of the
/// codes the slot carries: the normal's are the first three, and the t adds
/// the last two.
pub const DIST_WITHHELD: [&str; 5] = [
    "no_weight",
    "min_weight",
    "degenerate_variance",
    "not_converged",
    "shape_unidentified",
];

/// The reasons a family's `withheld_reason_<t>` takes: [`DIST_WITHHELD`]'s
/// first three under the normal, all five under the t.
pub fn dist_withheld(family: rheon_core::DistFamily) -> &'static [&'static str] {
    match family {
        rheon_core::DistFamily::StudentT => &DIST_WITHHELD,
        _ => &DIST_WITHHELD[..3],
    }
}

/// The reason's code in [`DIST_WITHHELD`], as the slot carries it; NaN, a
/// null, while the target reports its fit.
fn code(reason: Option<DistWithheld>) -> f64 {
    match reason {
        None => f64::NAN,
        Some(r) => DIST_WITHHELD
            .iter()
            .position(|n| *n == r.name())
            .map_or(f64::NAN, |i| i as f64),
    }
}

/// [`DistFit`] with its reason slots: the state is the model's own, and it
/// serializes as the model alone.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct DistFitRows {
    model: DistFit,
    /// The reasons read before the row, one a target: scratch kept so a
    /// row allocates nothing, never state.
    #[serde(skip)]
    reasons: Vec<f64>,
}

impl PartialEq for DistFitRows {
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model
    }
}

impl DistFitRows {
    pub fn new(model: DistFit) -> Self {
        Self {
            reasons: Vec::with_capacity(model.cfg().n_targets),
            model,
        }
    }

    pub fn cfg(&self) -> &DistFitCfg {
        self.model.cfg()
    }

    pub fn n_eff(&self) -> f64 {
        self.model.n_eff()
    }

    fn read_reasons(&mut self) {
        self.reasons.clear();
        let m = &self.model;
        self.reasons
            .extend((0..m.cfg().n_targets).map(|j| code(m.withheld(j))));
    }

    /// Widen the model's `pred`, `per` slots a target, to `per + 1`, each
    /// target's reason last in its group. A refused step's empty `pred` is
    /// left empty.
    fn interleave(pred: &mut Vec<f64>, reasons: &[f64]) {
        let t = reasons.len();
        if t == 0 || pred.is_empty() || !pred.len().is_multiple_of(t) {
            return;
        }
        let per = pred.len() / t;
        pred.resize(t * (per + 1), f64::NAN);
        // From the last target back, so no slot is overwritten before it
        // moves.
        for j in (0..t).rev() {
            pred.copy_within(j * per..(j + 1) * per, j * (per + 1));
            pred[j * (per + 1) + per] = reasons[j];
        }
    }
}

impl OnlineModel for DistFitRows {
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        self.model.target_n_eff_into(out)
    }

    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        self.read_reasons();
        self.model.step_into(x, y, d_clock, weight, out);
        Self::interleave(&mut out.pred, &self.reasons);
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        self.predict_with(x, &[], d_clock)
    }

    fn predict_with(&self, x: &[f64], y: &[Option<f64>], d_clock: f64) -> Step {
        let mut out = self.model.predict_with(x, y, d_clock);
        let m = &self.model;
        let reasons: Vec<f64> = (0..m.cfg().n_targets)
            .map(|j| code(m.withheld(j)))
            .collect();
        Self::interleave(&mut out.pred, &reasons);
        out
    }

    fn state(&self) -> State {
        self.model.state()
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        DistFit::restore(s).map(Self::new)
    }

    fn n_targets(&self) -> usize {
        self.model.n_targets()
    }

    fn n_features(&self) -> usize {
        self.model.n_features()
    }

    fn n_outputs(&self) -> usize {
        self.model.n_outputs() + self.model.n_targets()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_target_carries_its_reason_last_in_its_group() {
        let mut pred = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        DistFitRows::interleave(&mut pred, &[0.0, 2.0]);
        assert_eq!(pred, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 2.0]);
    }

    #[test]
    fn the_codes_name_the_core_reasons_in_order() {
        for (i, r) in [
            DistWithheld::NoWeight,
            DistWithheld::MinWeight,
            DistWithheld::DegenerateVariance,
            DistWithheld::NotConverged,
            DistWithheld::ShapeUnidentified,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(code(Some(r)), i as f64);
        }
        assert!(code(None).is_nan());
    }
}
