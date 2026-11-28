//! `Dependence`: online nonlinear dependence between every feature and every
//! target — binned mutual information (with optional Miller-Madow bias
//! correction) and windowed rank correlations (Spearman's rho, Kendall's
//! tau-b, and Chatterjee's xi).
//!
//! Every pair is a feature `j` (of `k`) against a target `t` (of `T`), as
//! [`crate::Marginal`] pairs them. All outputs are read **before** the current
//! row is learned (out-of-sample by construction).
//!
//! # Measures
//!
//! - **Mutual information (`mi`)**: per pair a joint histogram of weights
//!   `n_ab` over feature bins `0..b_x` and target bins `0..b_y` (edges fixed
//!   from `bin_edges`/`target_bin_edges`, or learned once from `bin_warm_rows`
//!   or `bin_warm_span` via [`crate::edges_from`] and held rows replayed).
//!   Read in nats with `libm::log` at read time only, using the scale-free
//!   ratio form:
//!
//!   ```text
//!   MI = sum_{n_ab > 0} (n_ab / N) * ln((n_ab / n_a) * (N / n_b))
//!   ```
//!
//!   Supports all-history (`half_life = inf`), exponential decay (`half_life`
//!   or `lam`, undecayed table against one group `scale` folded at `1e-50`),
//!   or an exact row/clock window (`window_rows` or `window_size`).
//! - **Miller-Madow (`mi_mm`)**: finite-sample bias correction subtracted from
//!   plug-in MI, for unit weights without decay:
//!
//!   ```text
//!   MI_MM = MI - (B_xy - B_x - B_y + 1) / (2 * N)
//!   ```
//!
//!   with `B_xy`, `B_x`, `B_y` the nonzero joint cells and margins. Undefined
//!   (`NaN`) whenever any positive weight in the pair's active sample is not
//!   `1.0`; refused beside decay.
//! - **Kendall's tau-b (`kendall`)**: exact incremental pair counters
//!   `S = n_c - n_d`, `n_0`, `n_1` (ties in `x`), `n_2` (ties in `y`) over a
//!   row or clock window (`O(W)` per row per pair, `O(1)` read):
//!
//!   ```text
//!   tau_b = S / sqrt((n_0 - n_1) * (n_0 - n_2))
//!   ```
//!
//! - **Spearman's rho (`spearman`)**: Pearson correlation of average ranks
//!   over the window's finite rows, refreshed every `rank_every_rows` learned
//!   rows (default `1`) or `rank_every` clock units and held between reads.
//! - **Chatterjee's xi (`xi`)**: Chatterjee (2021) eq. 1.2 (which reduces to
//!   eq. 1.1 with no ties in `y`), sorting pairs stably by `x` so ties in `x`
//!   break deterministically by arrival order (oldest first):
//!
//!   ```text
//!   xi_n = 1 - n * sum |r_{i+1} - r_i| / (2 * sum l_i * (n - l_i))
//!   ```
//!
//!   Refreshed on the same cadence as Spearman.
//!
//! # Internal slot layout
//!
//! [`Step::pred`] stores slots **target-outer**:
//! `slot(t, m_idx, j) = (t * n_measures + m_idx) * n_features + j`, so each
//! target owns a contiguous chunk of `n_measures * n_features` slots for
//! per-target `min_weight` gating in `rheon-polars`.

use serde::{Deserialize, Serialize};

use crate::clock_span::{DEFAULT_STATE_BYTES, HeldCap, Place, check_max_state_bytes, start_due};
use crate::model::{
    ModelState, OnlineModel, SameBits, State, StateError, Step, check_cfg, check_dims,
    check_min_weight, check_schema, damaged,
};
use crate::since::Since;
use crate::{BinRule, Decay, Stamp, WindowClosed};

pub(crate) mod mi;
pub(crate) mod ranks;
#[cfg(test)]
mod tests;

use mi::{BinRing, MiHold, MiScratch, MiTables, bin_of, check_edges};
pub use mi::{MAX_BINS as DEPENDENCE_MAX_BINS, NO_BIN as DEPENDENCE_NO_BIN};
use ranks::{
    HeldFloats, KendallCounters, RankScratch, ValueRing, gather_pair, kendall_slice,
    spearman_and_xi_from_sample,
};

/// Default number of bins per feature (and target when `target_bins` is unset).
pub const DEFAULT_BINS: usize = 16;

/// Default warm-up rows for learning MI bin edges from `(b_x, b_y)`:
/// `max(ceil(max(b_x, b_y) / rho^2), 5 * b_x * b_y)` with `rho = 1 / sqrt(62.5)`
/// (matching `marginal`'s 1D quantile-edge precision and Cochran's 5 expected
/// observations per 2D joint cell).
pub fn default_bin_warm_rows(b_x: usize, b_y: usize) -> usize {
    let edge_rows = crate::margbins::default_bin_warm_rows(b_x.max(b_y));
    let cell_rows = b_x.saturating_mul(b_y).saturating_mul(5);
    edge_rows.max(cell_rows).max(2)
}

/// One nonlinear dependence measure [`Dependence`] can compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependenceMeasure {
    /// Plug-in binned mutual information in nats.
    Mi,
    /// Miller-Madow bias-corrected binned mutual information in nats (unit weights, no decay).
    MiMm,
    /// Windowed Spearman rank correlation.
    Spearman,
    /// Windowed Kendall's tau-b rank correlation.
    Kendall,
    /// Windowed Chatterjee's xi rank dependence coefficient.
    Xi,
}

impl DependenceMeasure {
    /// The measure's spec word and field stem.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mi => "mi",
            Self::MiMm => "mi_mm",
            Self::Spearman => "spearman",
            Self::Kendall => "kendall",
            Self::Xi => "xi",
        }
    }

    pub fn is_mi(self) -> bool {
        matches!(self, Self::Mi | Self::MiMm)
    }

    pub fn is_rank(self) -> bool {
        matches!(self, Self::Spearman | Self::Kendall | Self::Xi)
    }

    pub fn is_periodic_rank(self) -> bool {
        matches!(self, Self::Spearman | Self::Xi)
    }
}

/// One `(feature, target)` pair's current dependence readings (for `ModelBank.dependence`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DependencePair {
    /// Target's accumulated weight (rows where the target was present; inside the window under one).
    pub n_eff: f64,
    pub mi: f64,
    pub mi_mm: f64,
    pub spearman: f64,
    pub kendall: f64,
    pub xi: f64,
}

/// Configuration for [`Dependence`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DependenceCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub measures: Vec<DependenceMeasure>,
    pub decay: Decay,
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub min_weight: f64,
    pub bins: usize,
    pub target_bins: usize,
    pub bin_rule: BinRule,
    pub bin_warm_rows: Option<usize>,
    pub bin_warm_span: Option<f64>,
    pub bin_edges: Option<Vec<Vec<f64>>>,
    pub target_bin_edges: Option<Vec<Vec<f64>>>,
    pub window_rows: Option<usize>,
    pub window_clock: Option<f64>,
    pub closed: WindowClosed,
    pub rank_every_rows: Option<usize>,
    pub rank_every_clock: Option<f64>,
    pub max_state_bytes: Option<u64>,
}

impl Default for DependenceCfg {
    fn default() -> Self {
        Self {
            n_features: 1,
            n_targets: 1,
            measures: vec![DependenceMeasure::Mi],
            decay: Decay::Halflife(f64::INFINITY),
            min_weight: 0.0,
            bins: DEFAULT_BINS,
            target_bins: DEFAULT_BINS,
            bin_rule: BinRule::Quantile,
            bin_warm_rows: None,
            bin_warm_span: None,
            bin_edges: None,
            target_bin_edges: None,
            window_rows: None,
            window_clock: None,
            closed: WindowClosed::Right,
            rank_every_rows: None,
            rank_every_clock: None,
            max_state_bytes: None,
        }
    }
}

impl DependenceCfg {
    pub fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES)
    }

    pub fn has_mi(&self) -> bool {
        self.measures.iter().any(|m| m.is_mi())
    }

    pub fn has_mi_mm(&self) -> bool {
        self.measures.contains(&DependenceMeasure::MiMm)
    }

    pub fn has_ranks(&self) -> bool {
        self.measures.iter().any(|m| m.is_rank())
    }

    pub fn has_kendall(&self) -> bool {
        self.measures.contains(&DependenceMeasure::Kendall)
    }

    pub fn has_spearman(&self) -> bool {
        self.measures.contains(&DependenceMeasure::Spearman)
    }

    pub fn has_xi(&self) -> bool {
        self.measures.contains(&DependenceMeasure::Xi)
    }

    pub fn has_periodic_ranks(&self) -> bool {
        self.measures.iter().any(|m| m.is_periodic_rank())
    }

    pub fn decays(&self) -> bool {
        match self.decay {
            Decay::Halflife(h) => h.is_finite(),
            Decay::Lam(l) => l < 1.0,
        }
    }

    pub fn windowed(&self) -> bool {
        self.window_rows.is_some() || self.window_clock.is_some()
    }

    pub fn edges_all_fixed(&self) -> bool {
        self.bin_edges.is_some() && self.target_bin_edges.is_some()
    }

    pub fn needs_warmup(&self) -> bool {
        self.has_mi() && !self.edges_all_fixed()
    }

    /// Effective warm-up row count when `needs_warmup()` and `bin_warm_span` is unset:
    /// explicit `bin_warm_rows`, or [`default_bin_warm_rows`] from `(b_x, b_y)`,
    /// shrunk to `window_rows.max(2)` under `window_rows` and clamped to fit
    /// inside `max_state_bytes` when a separate [`MiHold`] buffer is allocated.
    pub fn resolved_bin_warm_rows(&self) -> Option<usize> {
        if !self.needs_warmup() || self.bin_warm_span.is_some() {
            return None;
        }
        if let Some(explicit) = self.bin_warm_rows {
            return Some(explicit);
        }
        let bx = match &self.bin_edges {
            Some(e) => e.iter().map(|v| v.len() + 1).max().unwrap_or(2),
            None => self.bins,
        };
        let by = match &self.target_bin_edges {
            Some(e) => e.iter().map(|v| v.len() + 1).max().unwrap_or(2),
            None => self.target_bins,
        };
        let mut warm = default_bin_warm_rows(bx, by);
        if let Some(w) = self.window_rows {
            warm = w.clamp(2, warm);
        }
        let reuses_ring = self.has_ranks() && self.window_rows.is_some_and(|wr| warm <= wr);
        if !reuses_ring {
            let per_row = MiHold::bytes_for(
                1,
                self.n_features,
                self.n_targets,
                self.decays(),
                self.window_clock.is_some(),
            );
            let ring_bytes = match self.window_rows {
                Some(w) if self.has_ranks() => {
                    ValueRing::bytes_for(w, self.n_features, self.n_targets, false)
                }
                _ => 0,
            };
            let overhead = self
                .base_bytes()
                .saturating_add(self.fixed_rank_bytes())
                .saturating_add(ring_bytes);
            let cap = self.cap();
            if cap > overhead && per_row > 0 {
                let max_fit = usize::try_from((cap - overhead) / per_row).unwrap_or(usize::MAX);
                warm = warm.min(max_fit.max(2));
            }
        }
        Some(warm)
    }

    /// Whether warm-up rows can be read directly from `ValueRing` without a separate [`MiHold`].
    pub(crate) fn warmup_reuses_value_ring(&self) -> bool {
        if !self.needs_warmup() || !self.has_ranks() || self.bin_warm_span.is_some() {
            return false;
        }
        match (self.window_rows, self.resolved_bin_warm_rows()) {
            (Some(wr), Some(warm)) => warm <= wr,
            _ => false,
        }
    }

    /// Effective periodic rank row cadence when `has_periodic_ranks()` and no clock cadence is set.
    pub fn resolved_rank_every_rows(&self) -> Option<usize> {
        if !self.has_periodic_ranks() || self.rank_every_clock.is_some() {
            return None;
        }
        Some(self.rank_every_rows.unwrap_or(1))
    }

    pub fn has_dynamic_state(&self) -> bool {
        self.window_clock.is_some() || (self.needs_warmup() && self.bin_warm_span.is_some())
    }

    fn base_bytes(&self) -> u64 {
        std::mem::size_of::<Dependence>() as u64 + (self.n_targets as u64).saturating_mul(24)
    }

    fn max_mi_table_bytes(&self) -> u64 {
        if !self.has_mi() {
            return 0;
        }
        let xb: Vec<usize> = match &self.bin_edges {
            Some(e) => e.iter().map(|v| v.len() + 1).collect(),
            None => vec![self.bins; self.n_features],
        };
        let yb: Vec<usize> = match &self.target_bin_edges {
            Some(e) => e.iter().map(|v| v.len() + 1).collect(),
            None => vec![self.target_bins; self.n_targets],
        };
        MiTables::max_bytes_for(self.n_features, self.n_targets, &xb, &yb, self.windowed())
    }

    fn fixed_rank_bytes(&self) -> u64 {
        let mut b = 0u64;
        let pairs = (self.n_features as u64).saturating_mul(self.n_targets as u64);
        if self.has_kendall() {
            b = b.saturating_add(KendallCounters::bytes(self.n_features, self.n_targets));
        }
        if self.has_spearman() {
            b = b.saturating_add(pairs.saturating_mul(8));
        }
        if self.has_xi() {
            b = b.saturating_add(pairs.saturating_mul(8));
        }
        b
    }

    /// Bytes fixed by the settings (checked up front in `validate`).
    pub fn fixed_bytes(&self) -> u64 {
        let base = self.base_bytes().saturating_add(self.fixed_rank_bytes());
        let mi_steady = self.max_mi_table_bytes();
        let ring = match self.window_rows {
            Some(w) if self.has_ranks() => {
                ValueRing::bytes_for(w, self.n_features, self.n_targets, false)
            }
            Some(w) if self.has_mi() => {
                BinRing::bytes_for(w, self.n_features, self.n_targets, false)
            }
            _ => 0,
        };
        let hold = if self.needs_warmup() && !self.warmup_reuses_value_ring() {
            self.resolved_bin_warm_rows().map_or(0, |r| {
                MiHold::bytes_for(
                    r,
                    self.n_features,
                    self.n_targets,
                    self.decays(),
                    self.window_clock.is_some(),
                )
            })
        } else {
            0
        };
        // Steady state after warm-up vs state while warm-up holds rows.
        let steady = base.saturating_add(mi_steady).saturating_add(ring);
        let during_warmup = base
            .saturating_add(if self.has_ranks() { ring } else { 0 })
            .saturating_add(hold);
        steady.max(during_warmup)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(check_dims("dependence", self.n_features, self.n_targets));
        if self.measures.is_empty() {
            no.refuse(
                "dependence: measures must list at least one measure, but it is empty. Choose \
                 one or more of \"mi\", \"mi_mm\", \"spearman\", \"kendall\" and \"xi\".",
            );
        } else {
            let mut seen = Vec::with_capacity(self.measures.len());
            for &m in &self.measures {
                if seen.contains(&m) {
                    no.refuse(format!(
                        "dependence: measures lists {:?} more than once. List each measure once.",
                        m.as_str()
                    ));
                    break;
                }
                seen.push(m);
            }
        }
        let decay_ok = no.take(self.decay.check().map_err(|e| format!("dependence: {e}")));
        no.take(check_min_weight("dependence", self.min_weight));
        no.stop_here()?;

        let decays = decay_ok && self.decays();
        let windowed = self.windowed();

        if self.window_rows == Some(0) {
            no.refuse(
                "dependence: window_rows must be at least 1, but it is 0. Give the number of \
                 rows the window keeps, or leave it out.",
            );
        }
        if let Some(d) = self.window_clock
            && !(d.is_finite() && d > 0.0)
        {
            no.refuse(format!(
                "dependence: window_size must be finite and above 0, but it is {d}. Give the \
                 span of clock units the window keeps, or leave it out."
            ));
        }
        if self.window_rows.is_some() && self.window_clock.is_some() {
            no.refuse(
                "dependence: window_rows and window_size are each a window, and only one can \
                 decide which rows are kept. Give window_rows or window_size, not both.",
            );
        }
        if self.closed == WindowClosed::Both && self.window_clock.is_none() {
            no.refuse(
                "dependence: closed = \"both\" names the edge of window_size, which is not set. \
                 Give window_size, or leave closed out.",
            );
        }
        if windowed && decays {
            no.refuse(
                "dependence: a window forgets by dropping the oldest rows, so it cannot sit \
                 beside a decay, but half_life (or lam) decays too. Leave half_life out (or set \
                 it to inf), or leave out the window.",
            );
        }
        if self.has_ranks() && !windowed {
            no.refuse(
                "dependence: rank measures (\"spearman\", \"kendall\", \"xi\") need a window so \
                 the state stays bounded, but neither window_rows nor window_size is set. Set \
                 window_rows or window_size, or ask for \"mi\" alone.",
            );
        }
        if self.has_ranks() && decays {
            no.refuse(
                "dependence: rank measures (\"spearman\", \"kendall\", \"xi\") do not take a \
                 decay, but half_life (or lam) decays. Leave half_life out (or set it to inf), \
                 or ask for \"mi\" alone.",
            );
        }
        if self.has_mi_mm() && decays {
            no.refuse(
                "dependence: \"mi_mm\" (Miller-Madow) is defined for unit weights without decay, \
                 but half_life (or lam) decays. Leave half_life out (or set it to inf), or use \
                 \"mi\" instead.",
            );
        }

        if self.has_mi() {
            if let Some(e) = &self.bin_edges {
                no.take(check_edges("feature", "bin_edges", self.n_features, e));
            } else if self.bins < 2 || self.bins > DEPENDENCE_MAX_BINS {
                no.refuse(format!(
                    "dependence: bins must be between 2 and {DEPENDENCE_MAX_BINS}, but it is {}. \
                     Set bins to 2 or more.",
                    self.bins
                ));
            }
            if let Some(e) = &self.target_bin_edges {
                no.take(check_edges("target", "target_bin_edges", self.n_targets, e));
            } else if self.target_bins < 2 || self.target_bins > DEPENDENCE_MAX_BINS {
                no.refuse(format!(
                    "dependence: target_bins must be between 2 and {DEPENDENCE_MAX_BINS}, but it \
                     is {}. Set target_bins to 2 or more.",
                    self.target_bins
                ));
            }
            if self.edges_all_fixed()
                && (self.bin_warm_rows.is_some() || self.bin_warm_span.is_some())
            {
                no.refuse(
                    "dependence: bin_warm_rows and bin_warm_span learn bin edges from warm-up \
                     rows, but both bin_edges and target_bin_edges are already given. Leave out \
                     bin_warm_rows and bin_warm_span, or drop explicit edges.",
                );
            }
            if self.bin_warm_rows.is_some() && self.bin_warm_span.is_some() {
                no.refuse(
                    "dependence: bin_warm_rows and bin_warm_span both set the bin warm-up. Leave \
                     one of them out.",
                );
            }
            if let Some(r) = self.bin_warm_rows
                && r < 2
            {
                no.refuse(format!(
                    "dependence: bin_warm_rows must be at least 2 so edges can split the range, \
                     but it is {r}. Set bin_warm_rows to 2 or more."
                ));
            }
            if let Some(s) = self.bin_warm_span
                && !(s.is_finite() && s > 0.0)
            {
                no.refuse(format!(
                    "dependence: bin_warm_span must be finite and above 0, but it is {s}. Give \
                     the clock span held to learn bin edges, or leave it out."
                ));
            }
        } else if self.bin_edges.is_some()
            || self.target_bin_edges.is_some()
            || self.bin_warm_rows.is_some()
            || self.bin_warm_span.is_some()
        {
            no.refuse(
                "dependence: binning settings (bin_edges, target_bin_edges, bin_warm_rows, \
                 bin_warm_span) apply only when \"mi\" or \"mi_mm\" is in measures. Add \"mi\" \
                 to measures, or leave those settings out.",
            );
        }

        if self.rank_every_rows.is_some() && self.rank_every_clock.is_some() {
            no.refuse(
                "dependence: rank_every and rank_every_rows both set the rank refresh cadence. \
                 Leave one of them out.",
            );
        }
        if !self.has_periodic_ranks()
            && (self.rank_every_rows.is_some() || self.rank_every_clock.is_some())
        {
            no.refuse(
                "dependence: rank_every and rank_every_rows refresh \"spearman\" and \"xi\", \
                 neither of which is in measures. Add \"spearman\" or \"xi\", or leave out the \
                 rank cadence.",
            );
        }
        if self.rank_every_rows == Some(0) {
            no.refuse(
                "dependence: rank_every_rows must be at least 1, but it is 0. Set \
                 rank_every_rows to 1 to refresh on every row, or to a larger count.",
            );
        }
        if let Some(c) = self.rank_every_clock
            && !(c.is_finite() && c > 0.0)
        {
            no.refuse(format!(
                "dependence: rank_every must be finite and above 0, but it is {c}. Set \
                 rank_every to a positive clock span, or leave it out."
            ));
        }

        no.take(check_max_state_bytes("dependence", self.max_state_bytes));
        no.stop_here()?;

        let (fixed, cap) = (self.fixed_bytes(), self.cap());
        if fixed > cap {
            no.refuse(format!(
                "dependence: a group's state would take {fixed} bytes, past max_state_bytes \
                 ({cap}). Shorten window_rows or bin_warm_rows, use fewer bins or columns, or \
                 raise max_state_bytes."
            ));
        }
        no.finish()
    }
}

/// Online nonlinear dependence model: binned MI (+ Miller-Madow) and windowed
/// Spearman, Kendall tau-b, and Chatterjee xi.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependence {
    cfg: DependenceCfg,
    /// Shared weight across all learned rows (decayed, or inside the window).
    w_sum: f64,
    /// Per-target accumulated weight (rows where target `t` was present; inside window under one).
    target_w: Vec<f64>,
    /// Per-target count of rows currently in the window where target `t` was present.
    target_rows: Vec<u64>,
    /// Per-target count of rows in the active sample/window with positive weight `!= 1.0`.
    target_non_unit: Vec<u64>,
    /// Total positive-weight rows learned so far.
    rows_learned: u64,
    /// Joint MI tables (`Some` once edges are fixed; `None` while warming up or when `!has_mi()`).
    mi_tables: Option<MiTables>,
    /// Warm-up row hold (`Some` while learning edges without reusing `ValueRing`).
    mi_hold: Option<MiHold>,
    /// Compact bin ring (used when `has_mi() && !has_ranks() && windowed()`).
    bin_ring: Option<BinRing>,
    /// Value ring (used when `has_ranks() && windowed()`).
    val_ring: Option<ValueRing>,
    /// Incremental Kendall tau-b counters (`Some` when `has_kendall()`).
    kendall: Option<KendallCounters>,
    /// Held Spearman values, target-outer `t * k + j` (empty when `!has_spearman()`).
    held_spearman: HeldFloats,
    /// Held Chatterjee xi values, target-outer `t * k + j` (empty when `!has_xi()`).
    held_xi: HeldFloats,
    /// Learned rows since the last periodic rank refresh (for `rank_every_rows`).
    rows_since_rank: usize,
    /// Clock cadence tracker for `rank_every` (`Some` when `rank_every_clock.is_some()`).
    rank_since: Option<Since>,
    /// Exact clock place for `window_clock` or `bin_warm_span`.
    place: Option<Place>,
    /// Bytes a row would have taken the state to past `max_state_bytes`.
    cap_over: Option<u64>,
    /// Half-of-cap notice already raised for this group.
    cap_warned: bool,
    /// Half-of-cap notice waiting for the caller to take.
    cap_warning: Option<u64>,
    /// Reusable MI scratch buffers (never state).
    #[serde(skip)]
    mi_scratch: MiScratch,
    /// Reusable rank scratch buffers (never state).
    #[serde(skip)]
    rank_scratch: RankScratch,
    /// Reusable per-row dense `y` buffer (`f64::NAN` for absent targets; never state).
    #[serde(skip)]
    y_buf: Vec<f64>,
}

impl PartialEq for Dependence {
    fn eq(&self, other: &Self) -> bool {
        self.cfg == other.cfg
            && self.w_sum.same_bits(&other.w_sum)
            && self.target_w.same_bits(&other.target_w)
            && self.target_rows == other.target_rows
            && self.target_non_unit == other.target_non_unit
            && self.rows_learned == other.rows_learned
            && self.mi_tables == other.mi_tables
            && self.mi_hold == other.mi_hold
            && self.bin_ring == other.bin_ring
            && self.val_ring == other.val_ring
            && self.kendall == other.kendall
            && self.held_spearman == other.held_spearman
            && self.held_xi == other.held_xi
            && self.rows_since_rank == other.rows_since_rank
            && self.rank_since == other.rank_since
            && self.place == other.place
            && self.cap_over == other.cap_over
            && self.cap_warned == other.cap_warned
            && self.cap_warning == other.cap_warning
    }
}

impl Dependence {
    pub fn new(cfg: DependenceCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.n_features;
        let nt = cfg.n_targets;
        let pairs = k * nt;
        let windowed = cfg.windowed();
        let clock_win = cfg.window_clock.is_some();

        let (mi_tables, mi_hold) = if !cfg.has_mi() {
            (None, None)
        } else if cfg.edges_all_fixed() {
            let xe = cfg.bin_edges.clone().expect("edges_all_fixed");
            let ye = cfg.target_bin_edges.clone().expect("edges_all_fixed");
            (Some(MiTables::new(xe, ye, windowed)), None)
        } else if cfg.warmup_reuses_value_ring() {
            (None, None)
        } else {
            let cap_rows = cfg.resolved_bin_warm_rows().unwrap_or(0);
            (
                None,
                Some(MiHold::new(k, nt, cap_rows, cfg.decays(), clock_win)),
            )
        };

        let val_ring = if cfg.has_ranks() && windowed {
            let cap_rows = cfg.window_rows.unwrap_or(0);
            Some(ValueRing::new(k, nt, cap_rows, clock_win))
        } else {
            None
        };
        let bin_ring = if cfg.has_mi() && !cfg.has_ranks() && windowed && mi_tables.is_some() {
            let cap_rows = cfg.window_rows.unwrap_or(0);
            Some(BinRing::new(k, nt, cap_rows, clock_win))
        } else {
            None
        };

        let kendall = cfg.has_kendall().then(|| KendallCounters::new(k, nt));
        let held_spearman = HeldFloats(if cfg.has_spearman() {
            vec![f64::NAN; pairs]
        } else {
            Vec::new()
        });
        let held_xi = HeldFloats(if cfg.has_xi() {
            vec![f64::NAN; pairs]
        } else {
            Vec::new()
        });
        let rank_since = cfg.rank_every_clock.map(|_| Since::default());
        let place = (cfg.window_clock.is_some()
            || (cfg.needs_warmup() && cfg.bin_warm_span.is_some()))
        .then(Place::default);

        Ok(Self {
            cfg,
            w_sum: 0.0,
            target_w: vec![0.0; nt],
            target_rows: vec![0; nt],
            target_non_unit: vec![0; nt],
            rows_learned: 0,
            mi_tables,
            mi_hold,
            bin_ring,
            val_ring,
            kendall,
            held_spearman,
            held_xi,
            rows_since_rank: 0,
            rank_since,
            place,
            cap_over: None,
            cap_warned: false,
            cap_warning: None,
            mi_scratch: MiScratch::default(),
            rank_scratch: RankScratch::default(),
            y_buf: Vec::with_capacity(nt),
        })
    }

    pub fn cfg(&self) -> &DependenceCfg {
        &self.cfg
    }

    /// Accumulated weight before the next row (inside the window under a window).
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// True while MI bin edges are still collecting warm-up rows.
    pub fn warming(&self) -> bool {
        self.cfg.has_mi() && self.mi_tables.is_none()
    }

    /// Last stamp held by a clock window or warm-up span (for fixture restoration).
    pub fn last_stamp(&self) -> Option<Stamp> {
        if let Some(vr) = &self.val_ring
            && let Some(&s) = vr.stamps.back()
        {
            return Some(s);
        }
        if let Some(br) = &self.bin_ring
            && let Some(&s) = br.stamps.back()
        {
            return Some(s);
        }
        if let Some(h) = &self.mi_hold
            && let Some(&s) = h.stamps.last()
        {
            return Some(s);
        }
        self.mi_hold.as_ref().and_then(|h| h.first_stamp)
    }

    /// Exact current state size in bytes (matching `DependenceCfg::fixed_bytes`).
    pub fn state_bytes(&self) -> u64 {
        let k = self.cfg.n_features;
        let nt = self.cfg.n_targets;
        let clock = self.cfg.window_clock.is_some();
        let mut b = self
            .cfg
            .base_bytes()
            .saturating_add(self.cfg.fixed_rank_bytes());
        if let Some(mt) = &self.mi_tables {
            b = b.saturating_add(mt.bytes());
        }
        if let Some(h) = &self.mi_hold {
            b = b.saturating_add(h.bytes(k, nt, self.cfg.decays(), clock));
        }
        if let Some(vr) = &self.val_ring {
            b = b.saturating_add(vr.bytes(k, nt, clock));
        }
        if let Some(br) = &self.bin_ring {
            b = b.saturating_add(br.bytes(k, nt, clock));
        }
        b
    }

    fn held_cap(&mut self) -> HeldCap<'_> {
        HeldCap {
            over: &mut self.cap_over,
            warned: &mut self.cap_warned,
            warning: &mut self.cap_warning,
        }
    }

    /// Number of front rows in the active ring that have left the clock window at `now`.
    fn expired_count(&self, now: Stamp) -> usize {
        let Some(span) = self.cfg.window_clock else {
            return 0;
        };
        let closed = self.cfg.closed;
        if let Some(vr) = &self.val_ring {
            vr.expired_count(now, span, closed)
        } else if let Some(br) = &self.bin_ring {
            br.expired_count(now, span, closed)
        } else {
            0
        }
    }

    /// Compute the exact post-step state bytes for an incoming positive-weight row
    /// at `now` BEFORE mutating any state (atomic [`HeldCap`] check).
    fn projected_bytes(&self, now: Option<Stamp>) -> u64 {
        let k = self.cfg.n_features;
        let nt = self.cfg.n_targets;
        let clock = self.cfg.window_clock.is_some();
        let evicted = now.map_or(0, |s| self.expired_count(s));

        let mut b = self
            .cfg
            .base_bytes()
            .saturating_add(self.cfg.fixed_rank_bytes());

        // Projected ring size after evicting expired rows (or 1 row at window_rows cap) + adding 1.
        let ring_add_len = |cur: usize| -> usize {
            let after_evict = cur.saturating_sub(evicted);
            let capped = match self.cfg.window_rows {
                Some(wr) if after_evict >= wr => wr - 1,
                _ => after_evict,
            };
            capped + 1
        };

        if let Some(vr) = &self.val_ring {
            b = b.saturating_add(ValueRing::bytes_for(ring_add_len(vr.len), k, nt, clock));
        }

        if let Some(mt) = &self.mi_tables {
            b = b.saturating_add(mt.bytes());
            if let Some(br) = &self.bin_ring {
                b = b.saturating_add(BinRing::bytes_for(ring_add_len(br.len), k, nt, clock));
            }
        } else if self.cfg.has_mi() {
            // Still warming up: check whether this row closes warm-up.
            let next_hold_rows = if let Some(h) = &self.mi_hold {
                h.rows + 1
            } else {
                // Reusing ValueRing for warm-up.
                (self.rows_learned as usize) + 1
            };
            let closes_warmup = if let Some(span) = self.cfg.bin_warm_span {
                let mut first = self.mi_hold.as_ref().and_then(|h| h.first_stamp);
                now.is_some_and(|s| start_due(&mut first, s, span, next_hold_rows, 2))
            } else {
                self.cfg
                    .resolved_bin_warm_rows()
                    .is_some_and(|target| next_hold_rows >= target)
            };
            if closes_warmup {
                // After freeze: MiHold is dropped, MiTables is built (and BinRing if MI-only windowed).
                b = b.saturating_add(self.cfg.max_mi_table_bytes());
                if !self.cfg.has_ranks() && self.cfg.windowed() {
                    let surviving = if let (Some(span), Some(s), Some(h)) =
                        (self.cfg.window_clock, now, &self.mi_hold)
                    {
                        let closed = self.cfg.closed;
                        h.stamps
                            .iter()
                            .filter(|&&st| !closed.has_left(s, st, span))
                            .count()
                            + 1
                    } else {
                        match self.cfg.window_rows {
                            Some(wr) => next_hold_rows.min(wr),
                            None => 0,
                        }
                    };
                    b = b.saturating_add(BinRing::bytes_for(surviving, k, nt, clock));
                }
            } else if let Some(h) = &self.mi_hold {
                let _ = h;
                b = b.saturating_add(MiHold::bytes_for(
                    next_hold_rows,
                    k,
                    nt,
                    self.cfg.decays(),
                    clock,
                ));
            }
        }
        b
    }

    /// Read all outputs at `now` (`None`: state as it stands; `Some(stamp)`:
    /// clock window slid to `stamp`, non-mutating) into `out_pred` (target-outer
    /// layout: `(t * n_measures + m_idx) * n_features + j`), returning `n_eff`.
    fn read_into_with_scratch(
        &self,
        now: Option<Stamp>,
        fresh_ranks: bool,
        out_pred: &mut Vec<f64>,
        ms: &mut MiScratch,
        rs: &mut RankScratch,
    ) -> f64 {
        let k = self.cfg.n_features;
        let nt = self.cfg.n_targets;
        let nm = self.cfg.measures.len();
        let n_slots = nt * nm * k;
        out_pred.clear();
        out_pred.resize(n_slots, f64::NAN);

        let skip = now.map_or(0, |s| self.expired_count(s));
        let (live_w_sum, live_tw, live_non_unit) =
            self.live_weights(now, skip, &mut ms.live_tw, &mut ms.live_nu);

        // Compute MI and MI_MM once per pair if either is requested.
        let want_mi = self.cfg.has_mi();
        let want_mi_mm = self.cfg.has_mi_mm();
        let want_spearman = self.cfg.has_spearman();
        let want_kendall = self.cfg.has_kendall();
        let want_xi = self.cfg.has_xi();

        let mi_idx = self
            .cfg
            .measures
            .iter()
            .position(|&m| m == DependenceMeasure::Mi);
        let mi_mm_idx = self
            .cfg
            .measures
            .iter()
            .position(|&m| m == DependenceMeasure::MiMm);
        let sp_idx = self
            .cfg
            .measures
            .iter()
            .position(|&m| m == DependenceMeasure::Spearman);
        let kd_idx = self
            .cfg
            .measures
            .iter()
            .position(|&m| m == DependenceMeasure::Kendall);
        let xi_idx = self
            .cfg
            .measures
            .iter()
            .position(|&m| m == DependenceMeasure::Xi);

        let need_fresh_periodic =
            fresh_ranks || (skip > 0 && self.cfg.resolved_rank_every_rows() == Some(1));
        let window_emptied = skip > 0 && self.ring_len() <= skip;

        // Use separate borrow of `na`/`nb` via `ms` cleanly by reading `live_tw`/`live_non_unit`
        // directly when `skip == 0` or indexing copied lengths — wait: `live_tw` may borrow
        // `ms.live_tw`, so keep `na`/`nb` passed directly through `mi_from_slice`!
        let na = &mut ms.na;
        let nb = &mut ms.nb;
        for t in 0..nt {
            let tw = live_tw[t];
            if tw < self.cfg.min_weight {
                continue;
            }
            let base_t = t * nm * k;
            let unit_ok = !self.cfg.decays() && live_non_unit[t] == 0;

            for j in 0..k {
                // MI & MI_MM
                if want_mi && let Some(mt) = &self.mi_tables {
                    let (mi_v, mm_v) = if skip == 0 {
                        mt.read_pair(j, t, want_mi_mm && unit_ok, na, nb)
                    } else if window_emptied {
                        (f64::NAN, f64::NAN)
                    } else if let Some(vr) = &self.val_ring {
                        let mut scratch = MiScratch {
                            na: std::mem::take(na),
                            nb: std::mem::take(nb),
                            ..MiScratch::default()
                        };
                        let res = mt.read_pair_from_value_ring(
                            vr,
                            skip,
                            j,
                            t,
                            want_mi_mm && unit_ok,
                            &mut scratch,
                        );
                        *na = scratch.na;
                        *nb = scratch.nb;
                        res
                    } else if let Some(br) = &self.bin_ring {
                        let mut scratch = MiScratch {
                            na: std::mem::take(na),
                            nb: std::mem::take(nb),
                            ..MiScratch::default()
                        };
                        let res = mt.read_pair_from_bin_ring(
                            br,
                            skip,
                            j,
                            t,
                            want_mi_mm && unit_ok,
                            &mut scratch,
                        );
                        *na = scratch.na;
                        *nb = scratch.nb;
                        res
                    } else {
                        mt.read_pair(j, t, want_mi_mm && unit_ok, na, nb)
                    };
                    if let Some(m_i) = mi_idx {
                        out_pred[base_t + m_i * k + j] = mi_v;
                    }
                    if let Some(m_i) = mi_mm_idx {
                        out_pred[base_t + m_i * k + j] = if unit_ok { mm_v } else { f64::NAN };
                    }
                }

                // Kendall
                if want_kendall && let Some(kd) = &self.kendall {
                    let v = if skip == 0 {
                        kd.tau_b(j, t, k)
                    } else if window_emptied {
                        f64::NAN
                    } else if let Some(vr) = &self.val_ring {
                        kendall_slice(vr, skip, j, t, k, nt, rs)
                    } else {
                        f64::NAN
                    };
                    if let Some(m_i) = kd_idx {
                        out_pred[base_t + m_i * k + j] = v;
                    }
                }

                // Spearman & Chatterjee xi
                if want_spearman || want_xi {
                    let (sp_v, xi_v) = if window_emptied {
                        (f64::NAN, f64::NAN)
                    } else if need_fresh_periodic {
                        if let Some(vr) = &self.val_ring {
                            gather_pair(vr, skip, j, t, k, nt, rs);
                            spearman_and_xi_from_sample(want_spearman, want_xi, rs)
                        } else {
                            (f64::NAN, f64::NAN)
                        }
                    } else {
                        let p = t * k + j;
                        (
                            if want_spearman {
                                self.held_spearman.0[p]
                            } else {
                                f64::NAN
                            },
                            if want_xi { self.held_xi.0[p] } else { f64::NAN },
                        )
                    };
                    if let Some(m_i) = sp_idx {
                        out_pred[base_t + m_i * k + j] = sp_v;
                    }
                    if let Some(m_i) = xi_idx {
                        out_pred[base_t + m_i * k + j] = xi_v;
                    }
                }
            }
        }
        live_w_sum
    }

    fn ring_len(&self) -> usize {
        if let Some(vr) = &self.val_ring {
            vr.len
        } else if let Some(br) = &self.bin_ring {
            br.len
        } else {
            0
        }
    }

    /// `(w_sum, target_w, target_non_unit)` over the live rows (after skipping `skip`
    /// expired front rows of a clock window, or reading in-window rows of `mi_hold`
    /// while MI-only windowed warm-up is still collecting).
    fn live_weights<'a>(
        &'a self,
        now: Option<Stamp>,
        skip: usize,
        tw_buf: &'a mut Vec<f64>,
        nu_buf: &'a mut Vec<u64>,
    ) -> (f64, &'a [f64], &'a [u64]) {
        if let Some(h) = &self.mi_hold
            && self.val_ring.is_none()
            && self.cfg.windowed()
        {
            let nt = self.cfg.n_targets;
            let mut w_sum = 0.0;
            tw_buf.clear();
            tw_buf.resize(nt, 0.0);
            nu_buf.clear();
            nu_buf.resize(nt, 0);
            let start = match self.cfg.window_rows {
                Some(wr) => h.rows.saturating_sub(wr),
                None => 0,
            };
            for r in start..h.rows {
                if let (Some(span), Some(now_s)) = (self.cfg.window_clock, now)
                    && let Some(&st) = h.stamps.get(r)
                    && self.cfg.closed.has_left(now_s, st, span)
                {
                    continue;
                }
                let w = h.ws[r];
                w_sum += w;
                let non_unit = u64::from(w != 1.0);
                for t in 0..nt {
                    if h.ys[r * nt + t].is_finite() {
                        tw_buf[t] += w;
                        nu_buf[t] += non_unit;
                    }
                }
            }
            return (w_sum, tw_buf.as_slice(), nu_buf.as_slice());
        }
        if skip == 0 {
            return (
                self.w_sum,
                self.target_w.as_slice(),
                self.target_non_unit.as_slice(),
            );
        }
        let nt = self.cfg.n_targets;
        let mut w_sum = 0.0;
        tw_buf.clear();
        tw_buf.resize(nt, 0.0);
        nu_buf.clear();
        nu_buf.resize(nt, 0);
        if let Some(vr) = &self.val_ring {
            for r in skip..vr.len {
                let w = vr.ws[r];
                w_sum += w;
                let non_unit = u64::from(w != 1.0);
                for t in 0..nt {
                    if vr.ys[r * nt + t].is_finite() {
                        tw_buf[t] += w;
                        nu_buf[t] += non_unit;
                    }
                }
            }
        } else if let Some(br) = &self.bin_ring {
            for r in skip..br.len {
                let w = br.ws[r];
                w_sum += w;
                let non_unit = u64::from(w != 1.0);
                for t in 0..nt {
                    if br.yb[r * nt + t] != DEPENDENCE_NO_BIN {
                        tw_buf[t] += w;
                        nu_buf[t] += non_unit;
                    }
                }
            }
        }
        (w_sum, tw_buf.as_slice(), nu_buf.as_slice())
    }

    /// Recompute `w_sum`, `target_w`, `target_rows`, `target_non_unit` for an
    /// active MI-only windowed warm-up hold at `now` so `n_eff()` always equals
    /// the exact in-window weight.
    fn sync_warmup_window_weights(&mut self, now: Option<Stamp>) {
        let Some(h) = &self.mi_hold else {
            return;
        };
        if self.val_ring.is_some() || !self.cfg.windowed() {
            return;
        }
        let nt = self.cfg.n_targets;
        self.w_sum = 0.0;
        self.target_w.fill(0.0);
        self.target_rows.fill(0);
        self.target_non_unit.fill(0);
        let start = match self.cfg.window_rows {
            Some(wr) => h.rows.saturating_sub(wr),
            None => 0,
        };
        for r in start..h.rows {
            if let (Some(span), Some(now_s)) = (self.cfg.window_clock, now)
                && let Some(&st) = h.stamps.get(r)
                && self.cfg.closed.has_left(now_s, st, span)
            {
                continue;
            }
            let w = h.ws[r];
            self.w_sum += w;
            let non_unit = u64::from(w != 1.0);
            for t in 0..nt {
                if h.ys[r * nt + t].is_finite() {
                    self.target_w[t] += w;
                    self.target_rows[t] += 1;
                    self.target_non_unit[t] += non_unit;
                }
            }
        }
    }

    /// Read all `(feature j, target t)` pairs from the current state, computing
    /// fresh Spearman and Chatterjee xi directly from the window (for `ModelBank.dependence`).
    /// Returned in target-outer order (`t * n_features + j`).
    pub fn read_pairs_now(&self) -> Vec<DependencePair> {
        let k = self.cfg.n_features;
        let nt = self.cfg.n_targets;
        let nm = self.cfg.measures.len();
        let mut pred = Vec::new();
        let mut ms = MiScratch::default();
        let mut rs = RankScratch::default();
        self.read_into_with_scratch(None, true, &mut pred, &mut ms, &mut rs);

        let mi_idx = self
            .cfg
            .measures
            .iter()
            .position(|&m| m == DependenceMeasure::Mi);
        let mi_mm_idx = self
            .cfg
            .measures
            .iter()
            .position(|&m| m == DependenceMeasure::MiMm);
        let sp_idx = self
            .cfg
            .measures
            .iter()
            .position(|&m| m == DependenceMeasure::Spearman);
        let kd_idx = self
            .cfg
            .measures
            .iter()
            .position(|&m| m == DependenceMeasure::Kendall);
        let xi_idx = self
            .cfg
            .measures
            .iter()
            .position(|&m| m == DependenceMeasure::Xi);

        let mut out = Vec::with_capacity(k * nt);
        for (t, &n_eff) in self.target_w.iter().enumerate().take(nt) {
            let base_t = t * nm * k;
            for j in 0..k {
                let at = |idx: Option<usize>| idx.map_or(f64::NAN, |m| pred[base_t + m * k + j]);
                out.push(DependencePair {
                    n_eff,
                    mi: at(mi_idx),
                    mi_mm: at(mi_mm_idx),
                    spearman: at(sp_idx),
                    kendall: at(kd_idx),
                    xi: at(xi_idx),
                });
            }
        }
        out
    }

    /// [`OnlineModel::predict`] at a caller-supplied exact `stamp` (for `reads_at_stamp`).
    pub fn predict_at(&self, x: &[f64], d_clock: f64, stamp: Stamp) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let now = self.place.as_ref().map(|_| stamp);
        let mut out = Step::default();
        let mut ms = MiScratch::default();
        let mut rs = RankScratch::default();
        out.n_eff = self.read_into_with_scratch(now, false, &mut out.pred, &mut ms, &mut rs);
        out
    }

    /// Evict the oldest row from `val_ring` (and update Kendall, MI, weights).
    fn evict_val_ring_front(&mut self) {
        let k = self.cfg.n_features;
        let nt = self.cfg.n_targets;
        let Some(vr) = self.val_ring.as_mut() else {
            return;
        };
        let Some(w_old) = vr.pop_front(
            k,
            nt,
            &mut self.rank_scratch.evict_x,
            &mut self.rank_scratch.evict_y,
        ) else {
            return;
        };
        if let Some(kd) = self.kendall.as_mut() {
            kd.update_against_ring(
                vr,
                &self.rank_scratch.evict_x,
                &self.rank_scratch.evict_y,
                -1,
            );
        }
        if let Some(mt) = self.mi_tables.as_mut() {
            mt.bin_row(
                &self.rank_scratch.evict_x,
                &self.rank_scratch.evict_y,
                &mut self.mi_scratch.x_bins,
                &mut self.mi_scratch.y_bins,
            );
            self.mi_scratch.resum.clear();
            self.mi_scratch.resum.resize(k * nt, false);
            mt.remove_binned(
                &self.mi_scratch.x_bins,
                &self.mi_scratch.y_bins,
                w_old,
                &mut self.mi_scratch.resum,
            );
            for j in 0..k {
                for t in 0..nt {
                    if self.mi_scratch.resum[j * nt + t] {
                        mt.clear_pair(j, t);
                        let xe = mt.x_edges[j].clone();
                        let ye = mt.y_edges[t].clone();
                        for r in 0..vr.len {
                            let a = bin_of(&xe, vr.xs[r * k + j]);
                            let b = bin_of(&ye, vr.ys[r * nt + t]);
                            mt.resum_add_pair(j, t, a, b, vr.ws[r]);
                        }
                    }
                }
            }
        }
        let evict_y = std::mem::take(&mut self.rank_scratch.evict_y);
        self.evict_weights(w_old, &evict_y, true);
        self.rank_scratch.evict_y = evict_y;
    }

    /// Evict the oldest row from `bin_ring` (and update MI and weights).
    fn evict_bin_ring_front(&mut self) {
        let k = self.cfg.n_features;
        let nt = self.cfg.n_targets;
        let Some(br) = self.bin_ring.as_mut() else {
            return;
        };
        let Some(w_old) = br.pop_front(
            k,
            nt,
            &mut self.mi_scratch.x_bins,
            &mut self.mi_scratch.y_bins,
        ) else {
            return;
        };
        if let Some(mt) = self.mi_tables.as_mut() {
            self.mi_scratch.resum.clear();
            self.mi_scratch.resum.resize(k * nt, false);
            mt.remove_binned(
                &self.mi_scratch.x_bins,
                &self.mi_scratch.y_bins,
                w_old,
                &mut self.mi_scratch.resum,
            );
            for j in 0..k {
                for t in 0..nt {
                    if self.mi_scratch.resum[j * nt + t] {
                        mt.clear_pair(j, t);
                        for r in 0..br.len {
                            mt.resum_add_pair(j, t, br.xb[r * k + j], br.yb[r * nt + t], br.ws[r]);
                        }
                    }
                }
            }
        }
        // Track which targets were present via `y_bins != NO_BIN`, reusing `evict_y`.
        let mut evict_y = std::mem::take(&mut self.rank_scratch.evict_y);
        evict_y.clear();
        for &yb in self.mi_scratch.y_bins.iter().take(nt) {
            evict_y.push(if yb != DEPENDENCE_NO_BIN {
                0.0
            } else {
                f64::NAN
            });
        }
        self.evict_weights(w_old, &evict_y, false);
        self.rank_scratch.evict_y = evict_y;
    }

    /// Subtract an evicted row's weight from `w_sum`, `target_w`, `target_rows`,
    /// `target_non_unit`, resumming from the active ring on large cancellation.
    fn evict_weights(&mut self, w_old: f64, y_row: &[f64], from_val_ring: bool) {
        let nt = self.cfg.n_targets;
        let non_unit = u64::from(w_old != 1.0);
        let empty = self.ring_len() == 0;
        if empty {
            self.w_sum = 0.0;
            self.target_w.fill(0.0);
            self.target_rows.fill(0);
            self.target_non_unit.fill(0);
            return;
        }
        let rem = self.w_sum - w_old;
        if !rem.is_finite() || rem <= 0.0 || w_old > 1e8 * rem {
            self.w_sum = if from_val_ring {
                self.val_ring
                    .as_ref()
                    .map_or(0.0, |vr| vr.ws.iter().copied().sum())
            } else {
                self.bin_ring
                    .as_ref()
                    .map_or(0.0, |br| br.ws.iter().copied().sum())
            };
        } else {
            self.w_sum = rem;
        }
        for (t, &yt) in y_row.iter().enumerate().take(nt) {
            if !yt.is_finite() {
                continue;
            }
            self.target_rows[t] = self.target_rows[t].saturating_sub(1);
            self.target_non_unit[t] = self.target_non_unit[t].saturating_sub(non_unit);
            if self.target_rows[t] == 0 {
                self.target_w[t] = 0.0;
                continue;
            }
            let tr = self.target_w[t] - w_old;
            if !tr.is_finite() || tr <= 0.0 || w_old > 1e8 * tr {
                self.target_w[t] = if from_val_ring {
                    self.val_ring.as_ref().map_or(0.0, |vr| {
                        (0..vr.len)
                            .filter(|&r| vr.ys[r * nt + t].is_finite())
                            .map(|r| vr.ws[r])
                            .sum()
                    })
                } else {
                    self.bin_ring.as_ref().map_or(0.0, |br| {
                        (0..br.len)
                            .filter(|&r| br.yb[r * nt + t] != DEPENDENCE_NO_BIN)
                            .map(|r| br.ws[r])
                            .sum()
                    })
                };
            } else {
                self.target_w[t] = tr;
            }
        }
    }

    /// Evict every expired front row of a clock window at `now`.
    fn trim_clock_window(&mut self, now: Stamp) {
        let count = self.expired_count(now);
        for _ in 0..count {
            if self.val_ring.is_some() {
                self.evict_val_ring_front();
            } else if self.bin_ring.is_some() {
                self.evict_bin_ring_front();
            }
        }
        if count > 0 && self.ring_len() == 0 {
            self.held_spearman.0.fill(f64::NAN);
            self.held_xi.0.fill(f64::NAN);
        }
    }

    /// Refresh held Spearman and Chatterjee xi values from `val_ring`.
    fn refresh_periodic_ranks(&mut self) {
        let Some(vr) = &self.val_ring else {
            return;
        };
        let k = self.cfg.n_features;
        let nt = self.cfg.n_targets;
        let want_sp = self.cfg.has_spearman();
        let want_xi = self.cfg.has_xi();
        if !want_sp && !want_xi {
            return;
        }
        let rs = &mut self.rank_scratch;
        for t in 0..nt {
            for j in 0..k {
                gather_pair(vr, 0, j, t, k, nt, rs);
                let (sp, xi) = spearman_and_xi_from_sample(want_sp, want_xi, rs);
                let p = t * k + j;
                if want_sp {
                    self.held_spearman.0[p] = sp;
                }
                if want_xi {
                    self.held_xi.0[p] = xi;
                }
            }
        }
    }

    /// Freeze MI bin edges from the warm-up sample (either `mi_hold` or `val_ring`),
    /// build [`MiTables`], replay the relevant rows into it (and `bin_ring` if MI-only
    /// windowed), and drop `mi_hold`.
    fn freeze_mi(&mut self, now: Option<Stamp>) {
        let k = self.cfg.n_features;
        let nt = self.cfg.n_targets;
        let windowed = self.cfg.windowed();
        let clock_win = self.cfg.window_clock.is_some();

        let hold = self.mi_hold.take();

        let x_edges: Vec<Vec<f64>> = match &self.cfg.bin_edges {
            Some(e) => e.clone(),
            None => (0..k)
                .map(|j| {
                    let mut vals: Vec<(f64, f64)> = if let Some(h) = &hold {
                        (0..h.rows).map(|r| (h.xs[r * k + j], h.ws[r])).collect()
                    } else if let Some(vr) = &self.val_ring {
                        (0..vr.len).map(|r| (vr.xs[r * k + j], vr.ws[r])).collect()
                    } else {
                        Vec::new()
                    };
                    crate::edges_from(self.cfg.bin_rule, self.cfg.bins, &mut vals)
                })
                .collect(),
        };
        let y_edges: Vec<Vec<f64>> = match &self.cfg.target_bin_edges {
            Some(e) => e.clone(),
            None => (0..nt)
                .map(|t| {
                    let mut vals: Vec<(f64, f64)> = if let Some(h) = &hold {
                        (0..h.rows).map(|r| (h.ys[r * nt + t], h.ws[r])).collect()
                    } else if let Some(vr) = &self.val_ring {
                        (0..vr.len).map(|r| (vr.ys[r * nt + t], vr.ws[r])).collect()
                    } else {
                        Vec::new()
                    };
                    crate::edges_from(self.cfg.bin_rule, self.cfg.target_bins, &mut vals)
                })
                .collect(),
        };

        let mut mt = MiTables::new(x_edges, y_edges, windowed);
        let mut xb = Vec::with_capacity(k);
        let mut yb = Vec::with_capacity(nt);

        if let Some(vr) = &self.val_ring {
            // Replay every row currently inside the shared rank window.
            let mut x_row = vec![0.0; k];
            let mut y_row = vec![0.0; nt];
            for r in 0..vr.len {
                for (j, dst) in x_row.iter_mut().enumerate().take(k) {
                    *dst = vr.xs[r * k + j];
                }
                for (t, dst) in y_row.iter_mut().enumerate().take(nt) {
                    *dst = vr.ys[r * nt + t];
                }
                mt.bin_row(&x_row, &y_row, &mut xb, &mut yb);
                mt.add_binned(&xb, &yb, vr.ws[r]);
            }
        } else if let Some(h) = hold {
            if !windowed {
                // Decay or all-history: replay every held row in order with its decay.
                for r in 0..h.rows {
                    if !h.lams.is_empty() {
                        mt.decay(h.lams[r]);
                    }
                    mt.bin_row(
                        &h.xs[r * k..(r + 1) * k],
                        &h.ys[r * nt..(r + 1) * nt],
                        &mut xb,
                        &mut yb,
                    );
                    mt.add_binned(&xb, &yb, h.ws[r]);
                }
            } else {
                // MI-only windowed: build BinRing with the rows still inside the window
                // and reset w_sum / target_w / target_rows / target_non_unit to match exactly.
                let cap_rows = self.cfg.window_rows.unwrap_or(0);
                let mut br = BinRing::new(k, nt, cap_rows, clock_win);
                self.w_sum = 0.0;
                self.target_w.fill(0.0);
                self.target_rows.fill(0);
                self.target_non_unit.fill(0);

                let start = match self.cfg.window_rows {
                    Some(wr) => h.rows.saturating_sub(wr),
                    None => 0,
                };
                for r in start..h.rows {
                    let st = if clock_win { Some(h.stamps[r]) } else { None };
                    if let (Some(span), Some(now_s), Some(row_s)) = (self.cfg.window_clock, now, st)
                        && self.cfg.closed.has_left(now_s, row_s, span)
                    {
                        continue;
                    }
                    let w = h.ws[r];
                    let y_slice = &h.ys[r * nt..(r + 1) * nt];
                    mt.bin_row(&h.xs[r * k..(r + 1) * k], y_slice, &mut xb, &mut yb);
                    mt.add_binned(&xb, &yb, w);
                    br.push(&xb, &yb, w, st);
                    self.w_sum += w;
                    let nu = u64::from(w != 1.0);
                    for (t, &yt) in y_slice.iter().enumerate().take(nt) {
                        if yt.is_finite() {
                            self.target_w[t] += w;
                            self.target_rows[t] += 1;
                            self.target_non_unit[t] += nu;
                        }
                    }
                }
                self.bin_ring = Some(br);
            }
        }
        self.mi_tables = Some(mt);
    }

    fn has_valid_state(&self) -> bool {
        let k = self.cfg.n_features;
        let nt = self.cfg.n_targets;
        let pairs = k * nt;
        let windowed = self.cfg.windowed();
        let decays = self.cfg.decays();
        let clock_win = self.cfg.window_clock.is_some();

        if !(self.w_sum.is_finite() && self.w_sum >= 0.0)
            || self.target_w.len() != nt
            || self.target_rows.len() != nt
            || self.target_non_unit.len() != nt
            || !self.target_w.iter().all(|w| w.is_finite() && *w >= 0.0)
        {
            return false;
        }
        let expect_place =
            clock_win || (self.cfg.needs_warmup() && self.cfg.bin_warm_span.is_some());
        if self.place.is_some() != expect_place {
            return false;
        }

        // MI tables & warm-up hold.
        if !self.cfg.has_mi() {
            if self.mi_tables.is_some() || self.mi_hold.is_some() {
                return false;
            }
        } else if let Some(mt) = &self.mi_tables {
            if self.mi_hold.is_some() || !mt.has_shape(k, nt, windowed, decays) {
                return false;
            }
        } else {
            if self.cfg.edges_all_fixed() {
                return false;
            }
            if self.cfg.warmup_reuses_value_ring() {
                if self.mi_hold.is_some() {
                    return false;
                }
                if let Some(warm) = self.cfg.resolved_bin_warm_rows()
                    && (self.rows_learned as usize) >= warm
                {
                    return false;
                }
            } else {
                let Some(h) = &self.mi_hold else {
                    return false;
                };
                if !h.has_shape(k, nt, decays, clock_win) {
                    return false;
                }
                if let Some(warm) = self.cfg.resolved_bin_warm_rows()
                    && h.rows >= warm
                {
                    return false;
                }
            }
        }

        // Rings.
        if !windowed {
            if self.val_ring.is_some() || self.bin_ring.is_some() {
                return false;
            }
        } else if self.cfg.has_ranks() {
            if self.bin_ring.is_some() {
                return false;
            }
            let Some(vr) = &self.val_ring else {
                return false;
            };
            if !vr.has_shape(k, nt, clock_win, self.cfg.window_rows) {
                return false;
            }
        } else {
            if self.val_ring.is_some() {
                return false;
            }
            if self.mi_tables.is_some() {
                let Some(br) = &self.bin_ring else {
                    return false;
                };
                if !br.has_shape(
                    k,
                    nt,
                    clock_win,
                    self.cfg.window_rows,
                    self.mi_tables.as_ref(),
                ) {
                    return false;
                }
            } else if self.bin_ring.is_some() {
                return false;
            }
        }

        // Ranks.
        if self.cfg.has_kendall() {
            let Some(kd) = &self.kendall else {
                return false;
            };
            if !kd.has_shape(k, nt) {
                return false;
            }
        } else if self.kendall.is_some() {
            return false;
        }
        if self.held_spearman.0.len() != if self.cfg.has_spearman() { pairs } else { 0 } {
            return false;
        }
        if self.held_xi.0.len() != if self.cfg.has_xi() { pairs } else { 0 } {
            return false;
        }
        if self.rank_since.is_some() != self.cfg.rank_every_clock.is_some() {
            return false;
        }

        // STATE-3 cap check: a restored state cannot exceed its configured max_state_bytes.
        if self.state_bytes() > self.cfg.cap() {
            return false;
        }
        true
    }
}

impl OnlineModel for Dependence {
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        let k = self.cfg.n_features;
        let nt = self.cfg.n_targets;

        // Peek current clock stamp without mutating state yet.
        let now = self.place.as_ref().map(|p| p.peek(d_clock));

        // 1. Out-of-sample read before learning (hard rules 2 & 8).
        let mut ms = std::mem::take(&mut self.mi_scratch);
        let mut rs = std::mem::take(&mut self.rank_scratch);
        out.n_eff = self.read_into_with_scratch(now, false, &mut out.pred, &mut ms, &mut rs);
        out.extra = None;
        self.mi_scratch = ms;
        self.rank_scratch = rs;

        // 2. Zero-weight row handed directly to core: advance clock / decay only, learn nothing.
        if weight < f64::MIN_POSITIVE {
            if let Some(p) = self.place.as_mut() {
                let stepped = p.step(d_clock);
                if self.cfg.window_clock.is_some() {
                    self.trim_clock_window(stepped);
                    self.sync_warmup_window_weights(Some(stepped));
                }
            }
            if let Some(s) = self.rank_since.as_mut() {
                s.step(d_clock);
            }
            if self.cfg.decays() {
                let lam = self.cfg.decay.factor(d_clock);
                self.w_sum *= lam;
                if self.w_sum < f64::MIN_POSITIVE {
                    self.w_sum = 0.0;
                }
                for tw in &mut self.target_w {
                    *tw *= lam;
                    if *tw < f64::MIN_POSITIVE {
                        *tw = 0.0;
                    }
                }
                if let Some(mt) = self.mi_tables.as_mut() {
                    mt.decay(lam);
                }
                if let Some(h) = self.mi_hold.as_mut() {
                    h.pending_lam *= lam;
                    if h.pending_lam < f64::MIN_POSITIVE {
                        h.pending_lam = 0.0;
                    }
                }
            }
            return;
        }

        // 3. Atomic HeldCap check BEFORE any state mutation (PLAN 537).
        if self.cfg.has_dynamic_state() {
            let with_bytes = self.projected_bytes(now);
            let cap = self.cfg.cap();
            if !self.held_cap().admit(with_bytes, cap) {
                if let Some(p) = self.place.as_mut() {
                    p.discard_next();
                }
                if let Some(s) = self.rank_since.as_mut() {
                    s.discard_next();
                }
                return;
            }
        }

        // 4. Commit stamp and prepare dense target buffer.
        let now = self.place.as_mut().map(|p| p.step(d_clock));
        self.y_buf.clear();
        for t in 0..nt {
            self.y_buf
                .push(y.get(t).copied().flatten().unwrap_or(f64::NAN));
        }

        // 5. Evict expired rows from clock window (or oldest row at window_rows cap).
        if let Some(s) = now
            && self.cfg.window_clock.is_some()
        {
            self.trim_clock_window(s);
        }
        if let Some(wr) = self.cfg.window_rows {
            if self.val_ring.as_ref().is_some_and(|vr| vr.len >= wr) {
                self.evict_val_ring_front();
            } else if self.bin_ring.as_ref().is_some_and(|br| br.len >= wr) {
                self.evict_bin_ring_front();
            }
        }

        // 6. Apply decay (when decaying; mutually exclusive with windows).
        let lam = if self.cfg.decays() {
            let f = self.cfg.decay.factor(d_clock);
            self.w_sum *= f;
            if self.w_sum < f64::MIN_POSITIVE {
                self.w_sum = 0.0;
            }
            for tw in &mut self.target_w {
                *tw *= f;
                if *tw < f64::MIN_POSITIVE {
                    *tw = 0.0;
                }
            }
            if let Some(mt) = self.mi_tables.as_mut() {
                mt.decay(f);
            }
            f
        } else {
            1.0
        };

        // 7. Update shared & per-target weights and non-unit counts.
        // Note: when MI-only windowed is still warming up (`mi_hold` active and `bin_ring` none),
        // we still maintain `w_sum` and `target_w` over the held rows in-window, and `freeze_mi`
        // reconciles them exactly to the window on freeze.
        let non_unit = u64::from(weight != 1.0);
        self.w_sum += weight;
        for t in 0..nt {
            if self.y_buf[t].is_finite() {
                self.target_w[t] += weight;
                if self.cfg.windowed() {
                    self.target_rows[t] += 1;
                }
                self.target_non_unit[t] += non_unit;
            }
        }
        self.rows_learned += 1;

        // 8. Update ValueRing & Kendall counters (when ranks enabled).
        if self.val_ring.is_some() {
            let y_slice = &self.y_buf;
            if let (Some(kd), Some(vr)) = (self.kendall.as_mut(), self.val_ring.as_ref()) {
                kd.update_against_ring(vr, &x[..k], y_slice, 1);
            }
            if let Some(vr) = self.val_ring.as_mut() {
                vr.push(
                    &x[..k],
                    y_slice,
                    weight,
                    if self.cfg.window_clock.is_some() {
                        now
                    } else {
                        None
                    },
                );
            }
        }

        // 9. Update MI tables or warm-up hold.
        if self.cfg.has_mi() {
            if let Some(mt) = self.mi_tables.as_mut() {
                mt.bin_row(
                    &x[..k],
                    &self.y_buf,
                    &mut self.mi_scratch.x_bins,
                    &mut self.mi_scratch.y_bins,
                );
                mt.add_binned(&self.mi_scratch.x_bins, &self.mi_scratch.y_bins, weight);
                if let Some(br) = self.bin_ring.as_mut() {
                    br.push(
                        &self.mi_scratch.x_bins,
                        &self.mi_scratch.y_bins,
                        weight,
                        if self.cfg.window_clock.is_some() {
                            now
                        } else {
                            None
                        },
                    );
                }
            } else {
                // Still warming up.
                let mut freeze_now = false;
                if let Some(h) = self.mi_hold.as_mut() {
                    h.push(
                        &x[..k],
                        &self.y_buf,
                        weight,
                        self.cfg.decays().then_some(lam),
                        if self.cfg.window_clock.is_some() {
                            now
                        } else {
                            None
                        },
                    );
                    if let Some(span) = self.cfg.bin_warm_span {
                        let s = now.expect("place active under bin_warm_span");
                        freeze_now = start_due(&mut h.first_stamp, s, span, h.rows, 2);
                    } else if let Some(target) = self.cfg.resolved_bin_warm_rows() {
                        freeze_now = h.rows >= target;
                    }
                } else if let Some(target) = self.cfg.resolved_bin_warm_rows() {
                    // Warm-up reuses ValueRing.
                    freeze_now = (self.rows_learned as usize) >= target;
                }
                if freeze_now {
                    self.freeze_mi(now);
                } else {
                    self.sync_warmup_window_weights(now);
                }
            }
        }

        // 10. Refresh periodic rank measures (Spearman & xi) when cadence is due.
        // The first refresh happens as soon as 2 positive-weight rows exist, and restarts both
        // row and clock cadences from that row so `rank_every = D` and `rank_every_rows = D`
        // agree to the bit on every unit clock (row, numeric, or temporal).
        if self.cfg.has_periodic_ranks() {
            let due = if let Some(span) = self.cfg.rank_every_clock {
                let since = self
                    .rank_since
                    .as_mut()
                    .expect("rank_since active under rank_every_clock");
                since.step(d_clock);
                if self.rows_learned == 2 || (self.rows_learned > 2 && since.reached(span)) {
                    since.restart();
                    true
                } else {
                    false
                }
            } else {
                let every = self.cfg.resolved_rank_every_rows().unwrap_or(1);
                if self.rows_learned < 2 {
                    false
                } else if self.rows_learned == 2 {
                    self.rows_since_rank = 0;
                    true
                } else {
                    self.rows_since_rank += 1;
                    if self.rows_since_rank >= every {
                        self.rows_since_rank = 0;
                        true
                    } else {
                        false
                    }
                }
            };
            if due {
                self.refresh_periodic_ranks();
            }
        }

        // 11. Half-cap warning for dynamic state (PLAN 537).
        if self.cfg.has_dynamic_state() {
            let bytes = self.state_bytes();
            let cap = self.cfg.cap();
            self.held_cap().note(bytes, cap);
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let now = self.place.as_ref().map(|p| p.peek(d_clock));
        let mut out = Step::default();
        let mut ms = MiScratch::default();
        let mut rs = RankScratch::default();
        out.n_eff = self.read_into_with_scratch(now, false, &mut out.pred, &mut ms, &mut rs);
        out
    }

    fn stamp_next(&mut self, stamp: Stamp) {
        if let Some(p) = self.place.as_mut() {
            p.hand(stamp);
        }
        if let Some(s) = self.rank_since.as_mut() {
            s.stamp_next(stamp);
        }
    }

    fn stamp_next_on_rows(&mut self, stamp: Stamp) {
        if let Some(p) = self.place.as_mut() {
            p.hand(stamp);
        }
        if let Some(s) = self.rank_since.as_mut() {
            s.stamp_next_on_rows(stamp);
        }
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        let now = self
            .place
            .as_ref()
            .and_then(|p| self.cfg.window_clock.map(|_| p.peek(0.0)));
        let skip = now.map_or(0, |s| self.expired_count(s));
        if skip == 0 && !(self.mi_hold.is_some() && self.val_ring.is_none() && self.cfg.windowed())
        {
            out.extend_from_slice(&self.target_w);
        } else {
            let mut nu = Vec::new();
            let _ = self.live_weights(now, skip, out, &mut nu);
        }
        true
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.cap_warning.take().map(|b| (b, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.cap_over.map(|b| (b, self.cfg.cap()))
    }

    fn state(&self) -> State {
        State::new(ModelState::Dependence(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Dependence(m) => {
                check_cfg("dependence", m.cfg.validate())?;
                if !m.has_valid_state() {
                    return Err(damaged(
                        "dependence",
                        "its running sums or window have the wrong shape",
                    ));
                }
                Ok((**m).clone())
            }
            other => Err(StateError::WrongModel {
                expected: "dependence",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * self.cfg.measures.len() * self.cfg.n_features
    }
}
