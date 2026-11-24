//! The clock between two rows, for a window's edge and mass: [`Gap`] and
//! the helpers around it, moved out of `windows.rs` under the source cap.

use super::*;

/// `now` minus `last` in clock units: between two temporal values, taken in
/// integer nanoseconds and rounded once, as the clock takes its steps.
/// The clock between two rows of one stretch, for a window's edge: the
/// exact difference of their raw clocks in integer nanoseconds on a
/// temporal clock (`exact`), else of the clocks carried as bits in `off`
/// (a number clock's own value; the policy time without a clock column),
/// never of two origin-subtracted policy times (task 159, W3).
/// Two rounded policy times differ by rounding, and a row exactly one
/// window after another landed on either side of the edge (review R2, W1:
/// 0.4 − 0.1 is not 0.3 in a double). Against a window given as a duration
/// the comparison is between integers ([`KernelDef::window_ns`]); against
/// one given as a number the gap is converted to seconds with the same two
/// operations a duration's text is (`seconds_of_ns`), so a gap of exactly
/// one window still compares equal to it.
#[derive(Clone, Copy, Debug)]
pub(super) enum Gap {
    Ns(i64),
    Secs(f64),
    /// An integer clock's gap in the column's own units, exact (task 200).
    Int(i128),
}

/// What a row's `off` holds: a temporal clock's nanoseconds, an integer
/// clock's value (docs/PLAN.md task 200), or the bits of a double -- a float
/// clock's value, or the policy time without a clock column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum OffForm {
    Ns,
    Int,
    Bits,
}

#[inline]
pub(super) fn gap(form: OffForm, from_off: i64, to_off: i64) -> Gap {
    match form {
        OffForm::Ns => Gap::Ns(to_off.saturating_sub(from_off)),
        OffForm::Int => Gap::Int(i128::from(to_off) - i128::from(from_off)),
        OffForm::Bits => Gap::Secs(f64::from_bits(to_off as u64) - f64::from_bits(from_off as u64)),
    }
}

impl Gap {
    /// The gap against the kernel's window, by the comparison a model's
    /// window decides its edge with ([`Stamp::cmp_span_ns`], task 175): in
    /// integer nanoseconds against a window written as a duration, through
    /// `seconds_of_ns` against one written as a number, and on a number
    /// clock the one subtraction of the raw values. No answer (a NaN) is
    /// `Equal`.
    #[inline]
    pub(super) fn cmp_window(self, k: &KernelDef) -> Ordering {
        let w = k.window_size.expect("a windowed kernel");
        self.cmp_span(w, k.window_ns.map(i128::from))
    }

    /// The gap against zero, by the same comparison.
    #[inline]
    pub(super) fn cmp_zero(self) -> Ordering {
        self.cmp_span(0.0, Some(0))
    }

    /// The gap as the newer of two stamps, the older at zero.
    #[inline]
    pub(super) fn cmp_span(self, span: f64, span_ns: Option<i128>) -> Ordering {
        match self {
            // `Stamp::cmp_span_ns`'s two arms for these pairs, written out
            // where this is called a row at a time: integer nanoseconds
            // against a span in nanoseconds, and a number clock's
            // `(a - b) - (ra - rb)` of the pair `s, 0` with removed time 0.
            Gap::Ns(n) => match span_ns {
                Some(w) => i128::from(n).cmp(&w),
                None => Stamp::Ns(i128::from(n)).cmp_span_ns(Stamp::Ns(0), span, None),
            },
            Gap::Secs(s) => ((s - 0.0) - 0.0)
                .partial_cmp(&span)
                .unwrap_or(Ordering::Equal),
            // A gap below 2^53 is a double exactly, and compares with the
            // span as the exact comparison does (a NaN span is `Equal`).
            Gap::Int(n) if n.unsigned_abs() < 1 << 53 => {
                (n as f64).partial_cmp(&span).unwrap_or(Ordering::Equal)
            }
            Gap::Int(n) => Stamp::Int(n, 0.0).cmp_span_ns(Stamp::Int(0, 0.0), span, None),
        }
    }
}

/// `now` minus `last` in clock units: between two temporal values, taken
/// in integer nanoseconds and rounded once; between two integer ones, in
/// integers and rounded once (task 200).
pub(crate) fn elapsed(now: ClockValue, last: ClockValue) -> f64 {
    match (now, last) {
        (ClockValue::Ns(c), ClockValue::Ns(p)) => seconds_of_ns(i128::from(c) - i128::from(p)),
        (ClockValue::I64(c), ClockValue::I64(p)) => int_step_f64(c, p),
        (c, p) => match c.int_step(p) {
            Some(step) => step as f64,
            None => c.seconds() - p.seconds(),
        },
    }
}

/// An integer clock's step `c - p` as a double, rounded once: the 64-bit
/// difference where it fits, which converts with one instruction, the 128-bit
/// one where it does not, which is a library call. Both round the same
/// integer to the nearest double, ties to even.
#[inline(always)]
pub(super) fn int_step_f64(c: i64, p: i64) -> f64 {
    match c.checked_sub(p) {
        Some(d) => d as f64,
        None => (i128::from(c) - i128::from(p)) as f64,
    }
}

/// Whether `now` is after `last`: `elapsed(now, last) > 0.0`, the sign of the
/// step, which between two temporal values, or two integer ones, is the
/// integers' own.
#[inline]
pub(super) fn is_after(now: ClockValue, last: ClockValue) -> bool {
    match (now, last) {
        (ClockValue::Ns(c), ClockValue::Ns(p)) | (ClockValue::I64(c), ClockValue::I64(p)) => c > p,
        _ => elapsed(now, last) > 0.0,
    }
}

/// Whether `now` is more than `cap` clock units after `last`: on an
/// integer clock their difference in integers against the cap, exactly
/// (task 200), and as [`elapsed`] gives it otherwise.
#[inline]
pub(crate) fn past_cap(now: ClockValue, last: ClockValue, cap: f64) -> bool {
    match (now, last) {
        // Between two temporal values, a step short of the cap's whole
        // seconds is not past it, as the integers say, with no conversion to
        // seconds: a step of fewer whole seconds than `floor(cap)` converts
        // to at most `floor(cap)` (the conversion never rounds past the next
        // whole second), which is not more than `cap`. Any other step is
        // converted and compared, as `elapsed` gives it.
        (ClockValue::Ns(c), ClockValue::Ns(p)) => {
            let step = i128::from(c) - i128::from(p);
            if (1.0..4.0e9).contains(&cap)
                && step < i128::from(u32::try_from(cap as u64).unwrap_or(0)) * 1_000_000_000
            {
                return false;
            }
            seconds_of_ns(step) > cap
        }
        // Neither pair has an integer step: `elapsed` is the difference.
        (ClockValue::F64(c), ClockValue::F64(p)) => c - p > cap,
        // An integer clock's step below 2^53 is a double exactly, and a
        // double compares with the cap as the exact comparison does
        // (`cmp_int_f64`: a NaN cap is past by neither); a larger step takes
        // the exact comparison, out of line.
        (ClockValue::I64(c), ClockValue::I64(p)) => match c.checked_sub(p) {
            Some(d) if d.unsigned_abs() < 1 << 53 => d as f64 > cap,
            _ => past_cap_general(now, last, cap),
        },
        _ => past_cap_general(now, last, cap),
    }
}

/// [`past_cap`] for any other pair, out of line: the exact comparison of an
/// integer clock's step against the cap takes a conversion to 128 bits that
/// the rows of a float or temporal clock, taking the arms above, would
/// otherwise pay for on every row.
#[inline(never)]
pub(super) fn past_cap_general(now: ClockValue, last: ClockValue, cap: f64) -> bool {
    match now.int_step(last) {
        Some(step) => rheon_core::cmp_int_f64(step, cap) == Some(Ordering::Greater),
        None => elapsed(now, last) > cap,
    }
}

pub(super) fn refuse_backwards(
    adv: &ClockAdvance,
    seq: u64,
    prev: Option<ClockValue>,
    now: Option<ClockValue>,
) -> Result<(), Refusal> {
    match adv.backwards {
        Some(raw) => Err(Refusal::Backwards {
            seq,
            back: -raw,
            prev,
            now,
            min_backwards_jump: adv.disorder.map(|d| d.min_backwards_jump),
        }),
        None => Ok(()),
    }
}

/// `∫_a^b λ^(T - s) ds` for a backward kernel, `∫_a^b λ^(s - T) ds` for a
/// forward one: the decayed mass of `[a, b]` seen from `T`, `a <= b`, the
/// part of it on the far side of `T` none. Under a finite half-life it is
/// the interval's own mass, `M(b − a)`, discounted from its near end to
/// `T`: `λ^(T − b) M(b − a)` looking back, `λ^(a − T) M(b − a)` ahead. The
/// difference `M(T − a) − M(T − b)` it was cancelled: both masses round to
/// `h / ln 2` from about 54 half-lives on, and the interval weighed 0 where
/// it was the only one (review 2026-11-23, WIN-1). With no decay the mass
/// is the length, `M(T − a) − M(T − b)` as before.
#[inline]
pub(super) fn mass_between(k: &KernelDef, t: f64, a: f64, b: f64) -> f64 {
    if b <= a {
        return 0.0;
    }
    match k.direction {
        Direction::Backward if k.half_life.is_infinite() => k.mass(t - a) - k.mass(t - b),
        Direction::Forward if k.half_life.is_infinite() => k.mass(b - t) - k.mass(a - t),
        Direction::Backward => {
            let b = b.min(t);
            if b <= a {
                0.0
            } else {
                k.discount(t - b) * k.mass(b - a)
            }
        }
        Direction::Forward => {
            let a = a.max(t);
            if b <= a {
                0.0
            } else {
                k.discount(a - t) * k.mass(b - a)
            }
        }
    }
}

/// [`mass_between`] with both factors taken through `memo`: a window's cut
/// reads the same two distances for each operator of a kernel.
#[inline]
pub(super) fn mass_between_in(k: &KernelDef, memo: &mut DecayMemo, t: f64, a: f64, b: f64) -> f64 {
    if b <= a {
        return 0.0;
    }
    match k.direction {
        Direction::Backward if k.half_life.is_infinite() => {
            k.mass_cut_in(t - a, memo) - k.mass_far_in(t - b, memo)
        }
        Direction::Backward => {
            let b = b.min(t);
            if b <= a {
                0.0
            } else {
                k.discount_slot(t - b, &mut memo.disc_cut) * k.mass_cut_in(b - a, memo)
            }
        }
        Direction::Forward => mass_between(k, t, a, b),
    }
}

/// [`mass_between`] for a forward kernel's interval ending at the next row
/// `b`, starting at its own row `a = t`: one mass, of the distance to the next
/// row, through `memo`, which a regular clock asks again at the next row.
#[inline]
pub(super) fn mass_between_fill(
    k: &KernelDef,
    memo: &mut DecayMemo,
    t: f64,
    a: f64,
    b: f64,
) -> f64 {
    if b <= a {
        return 0.0;
    }
    k.mass_in(b - t, memo) - k.mass(a - t)
}

/// [`mass_between`] for a forward kernel's close: the open row's interval
/// `[a, b]` seen from `from`, the time the mean's sums are anchored at (the
/// first row in the window with a value, at or before `a`), so the two add
/// as they are; the mean is their ratio, the same from any time. Its
/// discount and mass are on slots of their own beside the one a row's
/// interval to the next row takes ([`mass_between_in`]), so each repeats
/// where the clock does. With no decay the mass is the length, taken as the
/// difference of the two distances from the row `t` being closed, as it was.
#[inline]
pub(super) fn mass_between_close(
    k: &KernelDef,
    memo: &mut DecayMemo,
    t: f64,
    from: f64,
    a: f64,
    b: f64,
) -> f64 {
    if b <= a {
        return 0.0;
    }
    if k.half_life.is_infinite() {
        return k.mass_slot(b - t, &mut memo.mass_far)
            - k.mass_slot(a - t, &mut memo.fwd().mass_open);
    }
    let a = a.max(from);
    if b <= a {
        return 0.0;
    }
    k.discount_slot(a - from, &mut memo.fwd().disc_mean) * k.mass_slot(b - a, &mut memo.mass_far)
}

/// [`mass_between`] for a backward kernel's interval ending at `t` -- a
/// value held from `a` to the row's own time -- with `memo` holding the mass
/// of the distance back to `a` that the last ask was for: where operators of
/// a kernel hold from the same time (they read one column, as a mean and a
/// spread of it do), it is taken once, and where a clock steps by the same
/// amount row after row, once for the stream. A forward kernel's is
/// [`mass_between`]'s.
#[inline]
pub(super) fn mass_held_to(k: &KernelDef, t: f64, a: f64, memo: &mut DecayMemo) -> f64 {
    // The interval is `[a, b]` with `b` the row's own time `t`, as `mass_between`
    // takes it, its far end's mass `M(t - b)` read where it is.
    let b = t;
    if b <= a {
        return 0.0;
    }
    match k.direction {
        Direction::Backward => k.mass_held_in(t - a, memo) - k.mass(t - b),
        Direction::Forward => mass_between(k, t, a, b),
    }
}
