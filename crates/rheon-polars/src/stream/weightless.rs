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
    /// The plan of a row of weight 0, and its place on the elapsed clock
    /// where an `embargo` or a label buffer reads it (`keep_elapsed`). The
    /// row is scored as `predict` scores a row: from the last row that
    /// counted, over its step read on a copy of the clock
    /// ([`ClockState::advance_scoring`]), a session change that would start
    /// a fresh stream or blend toward the long run read as such a copy of
    /// the model ([`Weightless`]). The stream's clock, its row count, the
    /// `coef` cadence and the fresh fold (`fresh`, a drift reset's) are
    /// read and never moved, so the next row steps from the last row that
    /// counted -- one step on a row clock, one gap under `gap_cap`, one
    /// session rule -- and no model, window, diagnostic or held-row buffer
    /// sees the row. Its clock, without a clock column, is the place the
    /// next row that counts takes: a row clock counts no such row.
    #[cold]
    #[inline(never)]
    pub(super) fn weightless_plan(
        &self,
        spec: &Spec,
        cfg: &rheon_core::ClockCfg,
        (clock_state, fresh): (&ClockState, Option<&ClockState>),
        (c, session): (Option<ClockValue>, Option<u64>),
        (ri, i, fed, accept): (usize, usize, u64, bool),
        keep_elapsed: bool,
    ) -> (RowPlan, Option<Stamp>) {
        let adv = clock_state.clone().advance_scoring(cfg, c, session);
        let d_fresh = fresh.map_or(0.0, |f| f.clone().advance_scoring(cfg, c, session).d_clock);
        let zero = if adv.reset || (adv.session_changed && spec.closes_on_session()) {
            Weightless::Fresh
        } else if adv.session_changed {
            Weightless::Blended
        } else {
            Weightless::AsIs
        };
        // Its place on the elapsed clock, which an `embargo` reads to release
        // the held rows whose wait it has seen out.
        let place = keep_elapsed
            .then(|| {
                clock_state
                    .clone()
                    .advance_stamped(cfg, &self.exact_caps, c, session, true)
                    .elapsed_stamp
            })
            .flatten();
        // Its own stamp, for a model whose read decides a window's edge.
        let stamp = spec
            .model
            .reads_at_stamp()
            .then(|| read_stamp(clock_state, cfg, &self.exact_caps, c, session))
            .flatten();
        let plan = RowPlan {
            ri,
            i,
            pending: usize::MAX,
            d_clock: adv.d_clock,
            d_fresh,
            elapsed_step: adv.d_clock,
            stamp,
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
        };
        (plan, place)
    }
}

/// The stamp a row only scored is read at, on a copy of the clock: the one
/// it would take were it learned, or, for a row the clock would refuse as a
/// step back, the last learned row's, where a read of it stands (a step of
/// 0). `None` before the first stamped row.
pub(super) fn read_stamp(
    clock: &ClockState,
    cfg: &rheon_core::ClockCfg,
    caps: &rheon_core::ExactCaps,
    c: Option<ClockValue>,
    session: Option<u64>,
) -> Option<Stamp> {
    let adv = clock.clone().advance_stamped(cfg, caps, c, session, true);
    if adv.backwards.is_none() {
        return adv.stamp;
    }
    clock
        .clone()
        .advance_stamped(cfg, caps, clock.last_clock(), session, true)
        .stamp
}
