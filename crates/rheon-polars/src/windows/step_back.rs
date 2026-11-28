//! Per-group step-back skip handling (`on_step_back = "skip"`) for [`Windows`].

use super::*;

pub(super) fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

/// Transient per-run stale-row counters (excluded from serialization and
/// equality checks so saved/loaded cores compare equal).
#[derive(Debug, Clone, Default)]
pub(super) struct TransientSkips {
    pub(super) rows: u64,
    pub(super) groups: Vec<usize>,
}

impl PartialEq for TransientSkips {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Windows {
    /// Record a stale row skipped under `on_step_back = "skip"` without
    /// advancing any clock or touching any window, cross, profile, or seasonal
    /// state. The row stays aligned 1-to-1 in the output with NaN (null)
    /// operator slots.
    pub(super) fn push_stale(&mut self, gi: usize, seq: u64) {
        self.next_seq += 1;
        for _ in 0..self.n_outputs {
            self.held_values.push_back(f64::NAN);
        }
        self.held_meta.push_back(Meta {
            open: 0,
            accept: false,
            drop: false,
        });
        self.stale_seqs.push_back(seq);
        self.groups[gi].skipped_step_back = self.groups[gi].skipped_step_back.saturating_add(1);
        self.skipped_run.rows = self.skipped_run.rows.saturating_add(1);
        if !self.skipped_run.groups.contains(&gi) {
            self.skipped_run.groups.push(gi);
        }
        self.advance_ready();
    }

    /// Take the transient `(rows_skipped, groups_affected)` summary accumulated
    /// since the last call, returning `None` when no stale rows were skipped.
    pub fn take_step_back_counts(&mut self) -> Option<(u64, usize)> {
        let rows = std::mem::take(&mut self.skipped_run.rows);
        let groups = std::mem::take(&mut self.skipped_run.groups).len();
        (rows > 0).then_some((rows, groups))
    }

    /// Total persisted stale rows skipped across all groups.
    pub fn total_skipped_step_back(&self) -> u64 {
        self.groups
            .iter()
            .fold(0u64, |acc, g| acc.saturating_add(g.skipped_step_back))
    }

    /// Persisted stale-row skip counts by group key (sorted by key; `None` for
    /// the null group).
    pub fn skipped_step_back_by_group(&self) -> Vec<(Option<String>, u64)> {
        let mut out: Vec<(Option<String>, u64)> = self
            .index
            .iter()
            .map(|(k, &gi)| (Some(k.clone()), self.groups[gi].skipped_step_back))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        if let Some(gi) = self.null_group {
            out.push((None, self.groups[gi].skipped_step_back));
        }
        out
    }

    /// Verify persisted `stale_seqs` and per-group skip counts on load (STATE-3).
    pub(super) fn stale_seqs_valid(&self) -> bool {
        if !self.clock_cfg.on_clock_reset.skips()
            && (!self.stale_seqs.is_empty() || self.groups.iter().any(|g| g.skipped_step_back > 0))
        {
            return false;
        }
        let mut prev: Option<u64> = None;
        for &seq in &self.stale_seqs {
            if seq < self.held_first || seq >= self.next_seq {
                return false;
            }
            if prev.is_some_and(|p| seq <= p) {
                return false;
            }
            let meta = &self.held_meta[(seq - self.held_first) as usize];
            if meta.open != 0 || meta.drop {
                return false;
            }
            prev = Some(seq);
        }
        true
    }
}
