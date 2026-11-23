//! A row of weight 0 in the schedule (hard rule 9; the user, 2026-11-21:
//! "Zero weight rows should not count in the clock the next row should
//! decay as if they were not present. This is always true on every model.";
//! docs/PLAN.md task 539).

use super::*;

/// Whether a row weighs 0, and if so which state it is scored by: the state
/// as it stands, a fresh stream's (a session change that starts one), or a
/// copy blended toward the long run (`session_shrink`) -- the three classes
/// `predict` scores a row by ([`Stream::predict_chunk`]). The models are
/// never touched: a fresh or blended state is a copy for the one row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Weightless {
    /// A row of positive weight, or one only replayed or scored.
    No,
    AsIs,
    Fresh,
    Blended,
}

impl Stream {
    /// The plan of a row of weight 0. The
    /// row is scored as `predict` scores a row: from the last row that
    /// counted, over its step read on a copy of the clock
    /// ([`ClockState::advance_scoring`]), a session change that would start
    /// a fresh stream or blend toward the long run read as such a copy of
    /// the model ([`Weightless`]). The stream's clock, its row count, the
    /// `coef` cadence and the fresh fold (`fresh`, a drift reset's) are
    /// read and never moved, so the next row steps from the last row that
    /// counted -- one step on a row clock, one gap under `gap_cap`, one
    /// session rule -- and no model, window, diagnostic or held-row buffer
    /// sees the row. Under an `embargo` or `label_clock` it releases no held
    /// row, as `predict` releases none: the live models stay what the
    /// stream without it holds (the user, 2026-11-23: "Score as predict";
    /// review 2026-11-23, STRM-1), and it is scored over the held rows'
    /// clock and its own step, as a held row is. Its clock, without a clock
    /// column, is the place the next row that counts takes: a row clock
    /// counts no such row.
    #[cold]
    #[inline(never)]
    pub(super) fn weightless_plan(
        &self,
        spec: &Spec,
        cfg: &rheon_core::ClockCfg,
        (clock_state, fresh): (&ClockState, Option<&ClockState>),
        (c, session): (Option<ClockValue>, Option<u64>),
        (ri, i, fed, accept): (usize, usize, u64, bool),
    ) -> RowPlan {
        let adv = clock_state.clone().advance_scoring(cfg, c, session);
        let d_fresh = fresh.map_or(0.0, |f| f.clone().advance_scoring(cfg, c, session).d_clock);
        let zero = if adv.reset || (adv.session_changed && spec.closes_on_session()) {
            Weightless::Fresh
        } else if adv.session_changed {
            Weightless::Blended
        } else {
            Weightless::AsIs
        };
        RowPlan {
            ri,
            i,
            pending: usize::MAX,
            d_clock: adv.d_clock,
            d_fresh,
            elapsed_step: adv.d_clock,
            stamp: None,
            clock: c.or(Some(ClockValue::F64(fed as f64))),
            reset: false,
            blend: false,
            session_changed: false,
            backwards: false,
            capped: false,
            accept,
            want_coef: false,
            emit: true,
            drift_ri: ri,
            learn: false,
            buffered: false,
            parks: false,
            w: 0.0,
            zero,
        }
    }
}
