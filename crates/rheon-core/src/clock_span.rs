//! The two rules every span of the clock that holds rows keeps (docs/PLAN.md
//! task 538): when a span that began at one row has run its length, and how
//! the rows it holds are checked as they come.
//!
//! **A span's length** is read off the rows' exact stamps
//! ([`Stamp::cmp_span`]): integer nanoseconds on a temporal clock, one
//! subtraction of raw values on a number clock, the count of rows that
//! counted on the row clock. A span that began at the row stamped `first`
//! has run at `now` once `now` is at least `span` clock units past it. The
//! row that reaches it is the span's last: a start or a refit over the
//! first `D` of the clock holds the rows from its first through the first
//! row at least `D` after it, so on the row clock, where each row is one
//! unit and a row of weight 0 is none, `D + 1` rows. `garch`'s `start_span`,
//! `expectile`'s and `glm`'s `refit_span` read it here.
//!
//! **The rows such a span holds** are the data's to number, so the state's
//! bytes are checked as each row is taken (docs/PLAN.md task 537): the first
//! row to take the state past half of its cap raises a warning, once, and a
//! row that would take it past the cap is not taken, nor any after it; the
//! caller refuses the run ([`crate::OnlineModel::state_over_cap`]). Every
//! model with such a span keeps the same three fields for it -- the bytes
//! past the cap, whether the warning was raised and the warning not yet
//! taken -- and moves them here, so no model words the rule its own way.

use std::cmp::Ordering;

use crate::Stamp;

/// Whether a span that began at the row stamped `first` has run its length
/// `span` (clock units) by the row stamped `now`: `now` at least `span` past
/// `first`, decided on the exact stamps (the module docs).
pub(crate) fn span_reached(now: Stamp, first: Stamp, span: f64) -> bool {
    now.cmp_span(first, span) != Ordering::Less
}

/// The run-time check of a state whose size the data decides (the module
/// docs), over a model's own fields.
pub(crate) struct HeldCap<'a> {
    /// The bytes a row would have taken the state to, past the cap: set
    /// once, and from then on no row is taken.
    pub over: &'a mut Option<u64>,
    /// The half-of-the-cap warning has been raised (once a group).
    pub warned: &'a mut bool,
    /// The warning raised and not yet taken by the caller.
    pub warning: &'a mut Option<u64>,
}

impl HeldCap<'_> {
    /// Whether a row that takes the state to `with` bytes may be taken under
    /// `cap`: not once a row has passed it, nor a row that would. A row
    /// refused here is recorded as the one past the cap.
    pub fn admit(&mut self, with: u64, cap: u64) -> bool {
        if self.over.is_some() {
            return false;
        }
        if with > cap {
            *self.over = Some(with);
            return false;
        }
        true
    }

    /// The state now holds `bytes`: past half of `cap` for the first time,
    /// the warning is raised.
    pub fn note(&mut self, bytes: u64, cap: u64) {
        if !*self.warned && 2 * bytes > cap {
            *self.warned = true;
            *self.warning = Some(bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both sides of the length: a span is run at exactly its length and
    /// past it, and not a nanosecond (or a row) short of it, on every form
    /// of stamp.
    #[test]
    fn a_span_has_run_at_its_length_and_not_before() {
        let s = 1_000_000_000_i128;
        assert!(span_reached(Stamp::Ns(5 * s), Stamp::Ns(2 * s), 3.0));
        assert!(span_reached(Stamp::Ns(5 * s + 1), Stamp::Ns(2 * s), 3.0));
        assert!(!span_reached(Stamp::Ns(5 * s - 1), Stamp::Ns(2 * s), 3.0));
        assert!(span_reached(Stamp::from(4.0), Stamp::from(1.0), 3.0));
        assert!(!span_reached(Stamp::from(3.0), Stamp::from(1.0), 3.0));
        // 0.1 + 0.2 is not 0.3 in doubles; the stamps' one subtraction is
        // what the edge reads, as the window operators do.
        assert!(span_reached(
            Stamp::from(0.4),
            Stamp::from(0.1),
            0.30000000000000004
        ));
        assert!(span_reached(Stamp::Int(7, 0.0), Stamp::Int(4, 0.0), 3.0));
        assert!(!span_reached(Stamp::Int(6, 0.0), Stamp::Int(4, 0.0), 3.0));
    }

    /// The cap: under it a row is taken, at it too, past it not, and after
    /// one row past it no row is; the warning comes once, past half.
    #[test]
    fn the_cap_warns_once_past_half_and_refuses_past_the_cap() {
        let (mut over, mut warned, mut warning) = (None, false, None);
        let mut cap = HeldCap {
            over: &mut over,
            warned: &mut warned,
            warning: &mut warning,
        };
        assert!(cap.admit(50, 100));
        cap.note(50, 100);
        assert_eq!(*cap.warning, None, "exactly half is not past it");
        assert!(cap.admit(51, 100));
        cap.note(51, 100);
        assert_eq!(cap.warning.take(), Some(51));
        cap.note(60, 100);
        assert_eq!(*cap.warning, None, "said once");
        assert!(cap.admit(100, 100), "the cap itself holds");
        assert!(!cap.admit(101, 100));
        assert_eq!(*cap.over, Some(101));
        assert!(!cap.admit(10, 100), "nothing after a row past the cap");
        assert_eq!(*cap.over, Some(101));
    }
}
