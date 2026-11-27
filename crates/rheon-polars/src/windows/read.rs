//! A forward kernel's open row filled into its queue, a backward window
//! read, and an operator's value from its sums: moved from `windows.rs`
//! unchanged (task 561), so that file stays under the source cap.

use std::cmp::Ordering;

use super::*;

/// A forward kernel's open row, once the next row at `end` has ended its
/// interval, written as a queue row into `row`: a mean weighs the held
/// value over `[tau, end)`, a sum counts the row's own value, and only a
/// row with its own value counts toward `min_samples`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fill_forward<const W: bool>(
    k: &KernelDef,
    ops: &[OpDef],
    members: &[usize],
    tau: f64,
    end: f64,
    x: &[f64],
    held: &[f64],
    row: &mut [f64],
    memo: &mut DecayMemo,
) {
    for (j, &o) in members.iter().enumerate() {
        let (v, h) = (x[j], held[j]);
        let n = f64::from(u8::from(!v.is_nan()));
        let acc = match ops[o].stat {
            Stat::Mean if !h.is_nan() => {
                let mass = mass_between_fill(k, memo, tau, tau, end);
                Acc {
                    s: mass * h,
                    w: mass,
                    n,
                    ..Acc::default()
                }
            }
            Stat::PointMean if !v.is_nan() => Acc {
                s: v,
                w: 1.0,
                n,
                ..Acc::default()
            },
            // A variance and moments have no forward kernel (`check_op`).
            Stat::Mean | Stat::PointMean | Stat::Var | Stat::Std | Stat::Moments => Acc::default(),
            Stat::Sum
            | Stat::Rate
            | Stat::Quantile
            | Stat::Max
            | Stat::Min
            | Stat::ExactQuantile
            | Stat::ExactRank => Acc {
                s: if v.is_nan() { 0.0 } else { v },
                w: 0.0,
                n,
                ..Acc::default()
            },
        };
        let b = j * row_w::<W>();
        row[b] = tau;
        // A quantile counts the row's own value, as a sum does, and its
        // sketch reads it here (`Queue::push`).
        row[b + 1] = if ops[o].stat.keeps_own() { v } else { h };
        acc.write(&mut row[b + 2..], W);
    }
}

/// Each operator's value of a backward window at stamp time `tau`, the
/// queue evicted to the window first, into `out`; NaN where null.
#[allow(clippy::too_many_arguments)]
pub(super) fn read_backward<const W: bool>(
    k: &KernelDef,
    ops: &[OpDef],
    members: &[usize],
    q: &mut Queue,
    scratch: &mut [f64],
    tau: f64,
    off: i64,
    form: OffForm,
    out: &mut [f64],
) {
    if k.window_size.is_some() {
        // The row at the far edge is in under "left" and "both". Ages are
        // the exact gap from the row to this one (`gap`), one form
        // everywhere, so a row exactly a window old lands on the same side
        // every time.
        let far = k.closed.far(Direction::Backward);
        q.evict::<W>(k, scratch, |_, _, t_off| {
            let age = gap(form, t_off, off).cmp_window(k);
            if far {
                age == Ordering::Greater
            } else {
                age != Ordering::Less
            }
        });
    }
    let span = k.window_size.map_or(tau, |w| w.min(tau));
    let edge = k.window_size.map_or(f64::NEG_INFINITY, |w| tau - w);
    for (j, &o) in members.iter().enumerate() {
        let op = &ops[o];
        let var = op.stat.is_var();
        if op.stat.is_extremum() {
            // The sums count a window's values; its part holds the extremum.
            let (x, n) = match k.window_size {
                Some(_) => (
                    q.ext(j).windowed(None),
                    q.sum_op::<W>(k, j).map_or(0.0, |s| s.0.n),
                ),
                None => q.ext(j).running(),
            };
            out[j] = if n < f64::from(op.min_samples) {
                f64::NAN
            } else {
                x
            };
            continue;
        }
        if op.stat == Stat::ExactQuantile {
            out[j] = if q.ord(j).len() < op.min_samples {
                f64::NAN
            } else {
                let p = op
                    .quantile
                    .expect("an exact quantile has a level (`check_op`)");
                let interp = op.interpolation.unwrap_or(Interp::Nearest);
                q.ord(j).quantile(p, interp, None)
            };
            continue;
        }
        if op.stat == Stat::ExactRank {
            // Each pending row at the stamp ranks its own value (`write_backward_ranks`).
            out[j] = f64::NAN;
            continue;
        }
        let Some((sum, at)) = q.sum_op::<W>(k, j) else {
            out[j] = empty_value(op);
            continue;
        };
        if op.stat == Stat::PointMean {
            out[j] = value_of(k, op, sum, span);
            continue;
        }
        if op.stat == Stat::Quantile {
            // The sums count the values; the sketches hold them.
            out[j] = if sum.n < f64::from(op.min_samples) {
                f64::NAN
            } else {
                let p = op.quantile.expect("a quantile has a level (`check_op`)");
                q.quantile(k, j, p, None).unwrap_or(f64::NAN)
            };
            continue;
        }
        // The oldest held value's interval may start before the window:
        // only the part inside counts, so a mean (and a variance) takes the
        // window less the rows up to that one, then the row's mass inside,
        // with no subtraction to cancel. Each operator's oldest valued row
        // is its own.
        let cut = match (k.window_size, op.stat.weighs_held()) {
            (Some(_), true) => q
                .find_oldest(j, |st, i| !st.x::<W>(i, j).is_nan())
                .map(|(p, in_front, i)| {
                    let st = q.stack(in_front);
                    // A row exactly a window old (exact, from the raw
                    // clocks, as its membership was decided) holds its
                    // value up to the window's edge and no further: no mass
                    // inside it, however `tau - w` rounds against its `tau`.
                    let on_edge = gap(form, st.off[i], off).cmp_window(k) == Ordering::Equal;
                    (p, st.start::<W>(i, j), st.end[i], st.x::<W>(i, j), on_edge)
                })
                .filter(|&(_, start, _, _, on_edge)| on_edge || start < edge),
            _ => None,
        };
        let a = match cut {
            Some((p, start, end, x, on_edge)) => {
                let mut a = q
                    .sum_after_op::<W>(k, p, j)
                    .map_or(Acc::default(), |(r, r_at)| {
                        r.scaled(k.discount_in(tau - r_at, &mut q.decay), var)
                    });
                let m = if on_edge {
                    0.0
                } else {
                    mass_between_in(k, &mut q.decay, tau, start.max(edge), end)
                };
                a = a.plus(Acc::held(x, m, var), var);
                a
            }
            // The sums are anchored at the newest row; seen from `tau`.
            None => sum.scaled(k.discount_in(tau - at, &mut q.decay), var),
        };
        out[j] = value_of(k, op, a, span);
    }
}

/// A window with no row: 0.0 for a sum with `min_samples` 0 (Polars'
/// default), null (NaN) for the rest.
pub(super) fn empty_value(op: &OpDef) -> f64 {
    if op.stat == Stat::Sum && op.min_samples == 0 {
        0.0
    } else {
        f64::NAN
    }
}

/// An operator's value from its accumulator and the window's span `t`.
pub(super) fn value_of(k: &KernelDef, op: &OpDef, a: Acc, t: f64) -> f64 {
    if a.n < f64::from(op.min_samples) {
        return f64::NAN;
    }
    match op.stat {
        Stat::Mean => {
            if a.w > 0.0 {
                a.s / a.w
            } else {
                f64::NAN
            }
        }
        Stat::PointMean => {
            if a.n > 0.0 {
                a.s
            } else {
                f64::NAN
            }
        }
        // `min_samples` 0 only: no value sums to 0.0, whatever the dust.
        Stat::Sum if a.n == 0.0 => 0.0,
        Stat::Sum => a.s,
        // Read from the queue's sketches/trees (`read_backward`, `close`), and
        // from the group's moments (`flush_in`).
        Stat::Quantile
        | Stat::Max
        | Stat::Min
        | Stat::Moments
        | Stat::ExactQuantile
        | Stat::ExactRank => f64::NAN,
        Stat::Rate => rate_of(a, k.clock_mass(t)),
        Stat::Var | Stat::Std => {
            if a.w <= 0.0 {
                return f64::NAN;
            }
            let biased = a.q / a.w;
            // Polars' correction for unequal weights, `V1² / (V1² − V2)`
            // times the biased variance, its denominator as `2 e`; none where
            // one row carries the window's weight (`e = 0`).
            let var = if op.bias {
                biased
            } else if a.e > 0.0 {
                a.w * a.w / (2.0 * a.e) * biased
            } else {
                f64::NAN
            };
            if op.stat == Stat::Std {
                var.sqrt()
            } else {
                var
            }
        }
    }
}

/// A rate: the decayed sum over the mass `m` of the window's span, null
/// where there is none.
#[inline]
pub(super) fn rate_of(a: Acc, m: f64) -> f64 {
    if m > 0.0 { a.s / m } else { f64::NAN }
}

/// Write a backward kernel's values for row `seq`, under each operator's
/// `partial` where the window reached before the stretch.
pub(super) fn write_backward(
    ops: &[OpDef],
    members: &[usize],
    seq: u64,
    vals: &[f64],
    complete: bool,
    held: Held<'_>,
) {
    let r = usize::try_from(seq - held.first).expect("a row is held");
    for (j, &o) in members.iter().enumerate() {
        let op = &ops[o];
        if !complete && op.partial == Partial::Drop {
            held.meta[r].drop = true;
        }
        if complete || op.partial == Partial::Keep {
            held.values[r * held.n_outputs + o] = vals[j];
        }
    }
}

/// Write backward exact-rank operators per pending row: at a shared stamp each
/// row ranks its own value against the stamp's full window (matching Polars'
/// `rolling_rank_by`).
pub(super) fn write_backward_ranks<const W: bool>(
    ops: &[OpDef],
    members: &[usize],
    q: &Queue,
    pending: &[u64],
    complete: bool,
    held: Held<'_>,
) {
    let m = pending.len();
    for (p_idx, &seq) in pending.iter().enumerate() {
        let r = usize::try_from(seq - held.first).expect("a row is held");
        let k_from_newest = m - 1 - p_idx;
        for (j, &o) in members.iter().enumerate() {
            let op = &ops[o];
            if op.stat != Stat::ExactRank {
                continue;
            }
            if complete || op.partial == Partial::Keep {
                let ord = q.ord(j);
                let val = if ord.len() < op.min_samples {
                    f64::NAN
                } else {
                    let x = q.newest_x::<W>(k_from_newest, j);
                    ord.rank(x, op.rank_method.unwrap_or(RankMethod::Average), None)
                };
                held.values[r * held.n_outputs + o] = val;
            }
        }
    }
}
