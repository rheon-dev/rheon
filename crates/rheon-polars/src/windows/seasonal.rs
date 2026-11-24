//! STL and MSTL behind an embargo (`stl`, `mstl`): on each row of a group,
//! a component of the seasonal-trend decomposition of the group's last `W`
//! learned rows, read `lag` learned rows back (`rheon_core::StlOperator`).
//!
//! **Learned rows.** A row is learned when its input is present; a missing
//! row pushes nothing and counts in no lag. The operators take no weight:
//! STL weighs every row of its window alike.
//!
//! **What a row reads.** Under `closed = "left"` (the default) the window
//! before the row, so the value reads no row at or after its own; under
//! `"right"` the window ending at the row, which reads the row's own value:
//! a column decomposed with `"right"` and used to predict itself leaks its
//! own target. Null until the window holds `W` learned rows.
//!
//! **Restarts.** A clock break (a gap past `gap_cap`, a session change), a
//! reset or a step back empties the group's ring, as every window operator
//! starts over: STL is positional, so a gap would shift every phase after it.
//!
//! **State.** One ring of `W` values a group: `8 W + 24` bytes, checked
//! against the operator's cap when the run is built, so a row never grows it.

use rheon_core::{
    Closed as RingClosed, Decomposition, Mstl, Stl, StlCfg, StlOperator, StlRing, check_state_bytes,
};
use serde::{Deserialize, Serialize};

use super::{Group, Held};
use crate::stream::usable;

/// The component an operator emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StlComponent {
    /// The seasonal of the `i`-th season, shortest period first.
    Seasonal(u32),
    Trend,
    Resid,
}

/// One STL or MSTL operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StlDef {
    /// `(period, seasonal span)` per season; one season is STL, several MSTL.
    pub seasons: Vec<(u32, u32)>,
    /// MSTL's passes over its seasons.
    pub iterate: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trend: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low_pass: Option<u32>,
    pub seasonal_deg: u8,
    pub trend_deg: u8,
    pub low_pass_deg: u8,
    pub seasonal_jump: u32,
    pub trend_jump: u32,
    pub low_pass_jump: u32,
    pub robust: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inner_iter: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outer_iter: Option<u32>,
    /// The window `W`, resolved: the embargo's `2 h + 1` when not given.
    pub window: u32,
    /// The lag, resolved: `h`, plus 1 under `"left"`, when not given.
    pub lag: u32,
    pub closed: RingClosed,
    pub component: StlComponent,
    pub input: usize,
    /// The most bytes one group's ring may hold.
    pub cap: u64,
}

impl StlDef {
    /// The operator's name: `stl` for one season, `mstl` for several.
    pub fn name(&self) -> &'static str {
        if self.seasons.len() == 1 {
            "stl"
        } else {
            "mstl"
        }
    }

    fn base(&self) -> StlCfg {
        let (period, seasonal) = self.seasons[0];
        StlCfg {
            period: period as usize,
            seasonal: seasonal as usize,
            trend: self.trend.map(|v| v as usize),
            low_pass: self.low_pass.map(|v| v as usize),
            seasonal_deg: self.seasonal_deg,
            trend_deg: self.trend_deg,
            low_pass_deg: self.low_pass_deg,
            seasonal_jump: self.seasonal_jump as usize,
            trend_jump: self.trend_jump as usize,
            low_pass_jump: self.low_pass_jump as usize,
            robust: self.robust,
            inner_iter: self.inner_iter.map(|v| v as usize),
            outer_iter: self.outer_iter.map(|v| v as usize),
        }
    }

    /// The decomposition these settings run.
    pub fn decomposition(&self) -> Result<Decomposition, String> {
        let name = self.name();
        let prefix = |e: String| format!("{name}: {e}");
        if self.seasons.len() == 1 {
            Ok(Decomposition::Stl(Stl::new(&self.base()).map_err(prefix)?))
        } else {
            let seasons: Vec<(usize, usize)> = self
                .seasons
                .iter()
                .map(|&(p, s)| (p as usize, s as usize))
                .collect();
            let m = Mstl::new(&seasons, &self.base(), self.iterate as usize).map_err(prefix)?;
            Ok(Decomposition::Mstl(m))
        }
    }

    /// The checked operator.
    pub fn operator(&self) -> Result<StlOperator, String> {
        StlOperator::new(
            self.decomposition()?,
            Some(self.window as usize),
            Some(self.lag as usize),
            self.closed,
        )
    }

    /// The bytes one group's ring holds: `8 W + 24`.
    pub fn bytes(&self) -> u64 {
        rheon_core::state_bytes(self.window as usize)
    }

    /// Every setting checked, the component among the seasons, and the ring
    /// within the cap.
    pub fn check(&self) -> Result<(), String> {
        let name = self.name();
        if self.seasons.is_empty() {
            return Err(format!("{name}: no season is given. Give a period."));
        }
        self.operator()?;
        if let StlComponent::Seasonal(i) = self.component
            && i as usize >= self.seasons.len()
        {
            return Err(format!(
                "{name}: seasonal component {i} of {} seasons. Give one of 0 to {}.",
                self.seasons.len(),
                self.seasons.len() - 1
            ));
        }
        check_state_bytes(name, self.window as usize, self.cap)
    }
}

/// One group's ring of one operator.
pub(super) type SeasState = StlRing;

/// Whether each group's rings fit its operators.
pub(super) fn fits(defs: &[StlDef], states: &[SeasState]) -> bool {
    states.is_empty()
        || (states.len() == defs.len()
            && states
                .iter()
                .zip(defs)
                .all(|(s, d)| s.window().len() <= d.window as usize))
}

/// Where the operators' outputs start among a row's.
pub(super) struct Out<'a> {
    pub defs: &'a [StlDef],
    pub first: usize,
}

/// One row of a group: each operator learns the row's input when present
/// and writes its component, or NaN while its window is not full. The
/// operator is built from its settings, checked when the run was: a fit
/// of `W` rows costs far more than the check.
pub(super) fn row(
    out: &Out<'_>,
    g: &mut Group,
    clock: f64,
    values: &[f64],
    seq: u64,
    held: &mut Held<'_>,
) {
    if g.seasonal.len() != out.defs.len() {
        g.seasonal = out
            .defs
            .iter()
            .map(|d| StlRing::new(d.window as usize))
            .collect();
    }
    for (j, (def, ring)) in out.defs.iter().zip(g.seasonal.iter_mut()).enumerate() {
        let op = def.operator().expect("checked when the run was built");
        let x = values[def.input];
        let x = usable(x).then_some(x);
        let v = op
            .step(ring, x, clock)
            .map_or(f64::NAN, |c| match def.component {
                StlComponent::Seasonal(i) => c.seasonal[i as usize],
                StlComponent::Trend => c.trend,
                StlComponent::Resid => c.resid,
            });
        let r = (seq - held.first) as usize;
        held.values[r * held.n_outputs + out.first + j] = v;
    }
}

#[cfg(test)]
#[path = "seasonal_tests.rs"]
mod tests;
