//! The core against a brute-force loop that knows the whole stream: it
//! steps the same `ClockState`s (the definition of the policy clock),
//! cuts the stream into each group's stretches between events, and
//! computes every operator from its definition, with no queue, no
//! monoid and no closing order.
use rheon_core::{OnClockReset, SessionGap};

use super::far_tests::mass_bf;
use super::*;

/// The integer short cuts in `past_cap` and `is_after` say what the
/// conversion to seconds says, for every step around the cap's whole
/// seconds and its edges, at caps from a second to the largest the short
/// cut takes, and past them.
#[test]
fn the_temporal_short_cuts_in_past_cap_and_is_after_are_the_conversions() {
    let base = -4_000_000_000_000_000_000i64;
    let caps = [
        0.0,
        0.5,
        1.0,
        1.5,
        59.999_999_999,
        60.0,
        300.0,
        300.000_000_001,
        86_400.0,
        1e7,
        8.9e9,
        9.0e9,
        1e12,
        f64::MAX,
    ];
    for cap in caps {
        let edge = (cap.min(9.2e9) * 1e9) as i64;
        for wiggle in (-3i64..=3).chain([-1_000_000_000, 1_000_000_000, -1, 1_000, -1_000]) {
            for step in [edge.saturating_add(wiggle), wiggle, wiggle.abs(), 0, 1] {
                let (now, last) = (
                    ClockValue::Ns(base.saturating_add(step)),
                    ClockValue::Ns(base),
                );
                let want = elapsed(now, last) > cap;
                assert_eq!(past_cap(now, last, cap), want, "{cap} {step}");
                let (back, fwd) = (
                    ClockValue::Ns(base),
                    ClockValue::Ns(base.saturating_add(step)),
                );
                let want = elapsed(back, fwd) > cap;
                assert_eq!(past_cap(back, fwd, cap), want, "{cap} -{step}");
                assert_eq!(is_after(now, last), elapsed(now, last) > 0.0, "{step}");
                assert_eq!(is_after(back, fwd), elapsed(back, fwd) > 0.0, "-{step}");
            }
        }
    }
    // The other forms: a number clock's difference, an integer clock's
    // step against the cap exactly, and a mixed pair as the numbers.
    let numbers = [
        0.0,
        -0.0,
        1.5,
        299.5,
        300.0,
        1e300,
        -1e300,
        f64::INFINITY,
        f64::NAN,
    ];
    for cap in [0.0, 1.0, 300.0, 1e300] {
        for a in numbers {
            for b in numbers {
                let (now, last) = (ClockValue::F64(a), ClockValue::F64(b));
                assert_eq!(past_cap(now, last, cap), elapsed(now, last) > cap);
                assert_eq!(is_after(now, last), elapsed(now, last) > 0.0);
            }
        }
        for (a, b) in [
            (0i64, 0i64),
            (301, 0),
            (300, 0),
            (i64::MAX, i64::MIN),
            (-5, 5),
        ] {
            let (now, last) = (ClockValue::I64(a), ClockValue::I64(b));
            let step = i128::from(a) - i128::from(b);
            let want = rheon_core::cmp_int_f64(step, cap) == Some(Ordering::Greater);
            assert_eq!(past_cap(now, last, cap), want, "{a} {b} {cap}");
        }
        let (now, last) = (ClockValue::F64(400.0), ClockValue::I64(0));
        assert_eq!(past_cap(now, last, cap), elapsed(now, last) > cap);
    }
}

/// [`long_windows_keep_their_bits`]'s digest, computed on the code
/// before task 270 (M-2), and moved once since: a window looking
/// ahead whose far edge is the input's last row is whole at the end of
/// the input where it was null.
const LONG_WINDOWS_DIGEST: u64 = 0x9224_8a97_1fbc_e295;

#[derive(Debug, Clone)]
struct Row {
    group: Option<String>,
    clock: Option<ClockValue>,
    session: Option<u64>,
    values: Vec<f64>,
    accept: bool,
}

/// Every output of every row, in input order.
#[derive(Debug, Clone, Default, PartialEq)]
struct Table {
    values: Vec<Vec<f64>>,
    drop: Vec<bool>,
}

impl Table {
    fn append(&mut self, e: Emitted, next: &mut u64) {
        assert_eq!(e.first_seq, *next, "rows leave in order, each once");
        *next += e.rows as u64;
        if self.values.is_empty() {
            self.values = vec![Vec::new(); e.values.len()];
        }
        for (a, b) in self.values.iter_mut().zip(e.values) {
            a.extend(b);
        }
        self.drop.extend(e.drop);
    }

    /// Bit for bit, NaN equal to NaN.
    fn same_bits(&self, other: &Table) -> bool {
        self.drop == other.drop
            && self.values.len() == other.values.len()
            && self.values.iter().zip(&other.values).all(|(a, b)| {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            })
    }

    fn close_to(&self, other: &Table, tol: f64) -> Result<(), (usize, usize, String)> {
        if self.drop != other.drop {
            let i = self
                .drop
                .iter()
                .zip(&other.drop)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Err((usize::MAX, i, "drop flags differ".into()));
        }
        for (o, (a, b)) in self.values.iter().zip(&other.values).enumerate() {
            if a.len() != b.len() {
                return Err((
                    o,
                    0,
                    format!("output {o}: {} rows against {}", a.len(), b.len()),
                ));
            }
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                if x.is_nan() != y.is_nan()
                    || !x.is_nan() && (x - y).abs() > tol * (1.0 + x.abs().max(y.abs()))
                {
                    return Err((o, i, format!("output {o} row {i}: {x} against {y}")));
                }
            }
        }
        Ok(())
    }
}

fn push(core: &mut Windows, r: &Row) -> Result<(), Refusal> {
    let group = core.group(r.group.as_deref());
    core.push(&RowIn {
        group,
        clock: r.clock,
        session: r.session,
        values: &r.values,
        accept: r.accept,
        lists: &[],
    })
}

/// The core, draining every `every` rows.
fn run(
    kernels: &[KernelDef],
    ops: &[OpDef],
    cfg: ClockCfg,
    rows: &[Row],
    every: usize,
) -> Result<Table, Refusal> {
    let mut core = Windows::new(kernels.to_vec(), ops.to_vec(), cfg).unwrap();
    core.set_grouped(rows.iter().any(|r| r.group.is_some()));
    let (mut t, mut next) = (Table::default(), 0);
    for (i, r) in rows.iter().enumerate() {
        push(&mut core, r)?;
        if (i + 1) % every == 0 {
            t.append(core.drain(), &mut next);
        }
        // Every state a stream passes through holds what a loader
        // checks (review 2026-10-06, PD1): `check` refuses no core this
        // module made, across every stream the brute force runs.
        core.check()
            .unwrap_or_else(|e| panic!("after row {i}: {e}"));
        pending_are_the_tail(&core).unwrap_or_else(|e| panic!("after row {i}: {e}"));
    }
    t.append(core.finish(), &mut next);
    core.check().unwrap_or_else(|e| panic!("finished: {e}"));
    assert_eq!(next, rows.len() as u64);
    Ok(t)
}

/// What [`Windows::flush_stamps_before`] walks from the tail on (task
/// 325): the groups with rows pending are the silent list's tail, each
/// at the stream's latest stamp, and no group off the list has any.
/// Held after every row of every stream [`run`] feeds -- gaps, cuts,
/// sessions, step backs and resets among them -- so a stream that broke
/// it, where a walk from the tail would leave a group's rows pending
/// that the head walk resolved, fails here by name.
fn pending_are_the_tail(core: &Windows) -> Result<(), String> {
    let latest = core.shared.last_clock();
    let (mut at, mut tail) = (core.head, false);
    while let Some(i) = at {
        let g = &core.groups[i];
        let pends = !g.pending.is_empty();
        if tail && !pends {
            return Err(format!("group {i} pends nothing after a group that pends"));
        }
        let current = matches!((g.stamp, latest), (Some(s), Some(l)) if elapsed(l, s) == 0.0);
        if pends && !current {
            return Err(format!("group {i} pends rows at an earlier stamp"));
        }
        tail |= pends;
        at = g.next_link;
    }
    if let Some(i) = core
        .groups
        .iter()
        .position(|g| !g.linked && !g.pending.is_empty())
    {
        return Err(format!("group {i} pends rows off the silent list"));
    }
    Ok(())
}

/// Each invariant [`Windows::check`] holds a loaded core to, broken
/// alone on a core fed a stream with groups, sessions, gaps and every
/// kernel -- so with rows held, waiting on forward windows and on the
/// next stamp, and groups in the silent list -- is refused by name
/// (review 2026-10-06, PD1).
#[test]
fn each_part_of_a_cores_invariants_is_checked_alone() {
    let (kernels, ops) = all_kernels();
    let rows = stream(9, 120, 3, true, true);
    let clock = cfg(8.0, Some(SessionGap::Gap(2.0)), None);
    let mut core = Windows::new(kernels, ops, clock).unwrap();
    core.set_grouped(true);
    for r in &rows[..77] {
        push(&mut core, r).unwrap();
    }
    core.drain_kept(3);
    core.check().unwrap();
    let is_fwd = |k: &KernelDef| k.direction == Direction::Forward;
    let fwd = core.kernels.iter().position(is_fwd).unwrap();
    let held_in = |f: fn(&Group) -> bool| core.groups.iter().position(f);
    let waiting = held_in(|g| !g.waiting.is_empty()).expect("a waiting row");
    let pending = held_in(|g| !g.pending.is_empty()).expect("a pending row");
    let (head, tail) = (core.head.expect("a silent list"), core.tail.unwrap());
    assert_ne!(head, tail, "two groups listed");
    let fit = "does not fit";
    type Damage<'a> = (&'a str, &'a dyn Fn(&mut Windows));
    let damage: [Damage; 24] = [
        ("names kernel", &|c| {
            Arc::make_mut(&mut c.ops)[0].kernel = 99
        }),
        ("operators are not", &|c| {
            Arc::make_mut(&mut c.members).swap(0, 1)
        }),
        ("outputs for", &|c| c.n_outputs += 1),
        ("forward kernels of", &|c| c.n_forward += 1),
        ("values held for", &|c| {
            c.held_values.pop_back();
        }),
        ("rows ready of", &|c| c.ready = c.held_meta.len() + 1),
        ("rows ready of", &|c| c.ready_kept += 1),
        ("before row", &|c| c.held_first += 1),
        (fit, &|c| c.groups[0].queues[0].n += 1),
        (fit, &|c| c.groups[0].queues[fwd].back.tau.push(0.0)),
        (fit, &|c| {
            c.groups[0].open_x[0].pop();
        }),
        (fit, &|c| {
            c.groups[waiting].closed[fwd] = c.groups[waiting].waiting.len() + 1;
        }),
        (fit, &|c| {
            c.groups[0].last_valued.pop();
        }),
        (fit, &|c| c.groups[waiting].waiting[0].seq = c.next_seq),
        (fit, &|c| {
            c.groups[pending].pending[0] = c.held_first.wrapping_sub(1);
        }),
        (fit, &|c| c.groups[head].last_raw = None),
        // A policy time past the largest double (review 2026-11-23, WIN-9).
        (fit, &|c| c.groups[0].tau = f64::INFINITY),
        ("open windows", &|c| c.held_meta[c.ready].open += 1),
        ("silent list", &|c| c.head = Some(c.groups.len())),
        ("silent list", &|c| c.groups[tail].next_link = Some(head)),
        // A list that walks whole but ends short of its tail (task 252).
        ("silent list", &|c| c.tail = c.head),
        ("one to one", &|c| {
            c.index.insert("stray".into(), 0);
        }),
        ("one to one", &|c| c.null_group = Some(c.groups.len())),
        // A group no key names (task 252): each key is distinct, but one
        // group is left without one.
        ("one to one", &|c| c.index.retain(|_, gi| *gi != 0)),
    ];
    for (says, f) in damage {
        let mut broken = core.clone();
        f(&mut broken);
        let err = broken
            .check()
            .err()
            .unwrap_or_else(|| panic!("{says}: passed"));
        assert!(err.contains(says), "{says}: {err}");
    }
}

/// `mass` at a half-life far past the window is the even kernel's, to
/// the series' next term: `1 − 2^(−t/h)` lost 6e-5 of it at `h/t = 1e12`
/// and all of it from 1e17, every mean null (task 159, W2). Where the
/// series is nowhere near, the closed form itself.
#[test]
fn mass_at_a_long_half_life_is_the_even_kernels() {
    let kernel = |h: f64| KernelDef {
        direction: Direction::Backward,
        half_life: h,
        window_size: Some(5.0),
        window_ns: None,
        closed: Closed::Right,
        since: None,
    };
    let ln2 = std::f64::consts::LN_2;
    for h in [1e9, 1e12, 1e17, 1e300, 1.5e308, f64::MAX] {
        let k = kernel(h);
        let want = 5.0 - 12.5 * ln2 / h;
        let got = k.clock_mass(5.0);
        assert!(
            (got - want).abs() <= 1e-12 * 5.0,
            "h = {h}: clock_mass {got} vs {want}"
        );
        let scaled = k.mass(5.0);
        let want_scaled = want * k.mass_scale();
        assert!(
            (scaled - want_scaled).abs() <= 1e-12 * want_scaled,
            "h = {h}: mass {scaled} vs {want_scaled}"
        );
    }
    let got = kernel(2.0).mass(3.0);
    let want = 2.0 / ln2 * (1.0 - 0.5f64.powf(1.5));
    // `(2.2 k + 4.4)·2⁻⁵²` for a libm `k` ulps off (docs/PLAN.md task
    // 273): `mass` is `−(h/ln 2)·exp_m1(−t ln 2/h)`, `exp_m1`'s `k` ulps
    // of 0.65 amplified 2.9 times (1.4 k), its argument's two
    // roundings (1.1), `h/ln 2`'s (0.9) and the product's (0.5); the
    // closed form's `powf` adds `k` ulps of 0.35, amplified alike (0.7
    // k), and its own three roundings (1.9). Held at `11·2⁻⁵²`, a libm
    // up to 3 ulps off (Windows' `ucrtbase` `expm1` was measured up to
    // 3 ulps off; the `k` of `rheon_core`'s oracle `LIBM`): measured 1
    // with Apple's, and 5 with Apple's moved by up to 2 ulps, where
    // `1e-15` (4.5), the bound before, failed.
    assert!(
        (got - want).abs() <= 11.0 * f64::EPSILON,
        "{got} vs {want}: {:e} of 2^-52",
        (got - want).abs() / f64::EPSILON
    );
}

/// A subnormal half-life passed `check`: `mass·x` underflowed to 0 and
/// a rate divided by that mass, so the mean came out 0 and the rate
/// infinite (task 159, W5).
#[test]
fn a_subnormal_half_life_is_refused() {
    for h in [5e-324, 1e-310] {
        let k = KernelDef {
            direction: Direction::Backward,
            half_life: h,
            window_size: Some(2.0),
            window_ns: None,
            closed: Closed::Right,
            since: None,
        };
        let err = k.check().expect_err("refused");
        assert!(err.contains("half_life must be at least"), "{err}");
    }
}

fn elapsed_bf(now: ClockValue, last: ClockValue) -> f64 {
    match (now, last) {
        (ClockValue::Ns(c), ClockValue::Ns(p)) => seconds_of_ns(i128::from(c) - i128::from(p)),
        (ClockValue::F64(c), ClockValue::F64(p)) => c - p,
        (ClockValue::I64(c), ClockValue::I64(p)) => (i128::from(c) - i128::from(p)) as f64,
        _ => unreachable!("a test stream keeps one kind of clock"),
    }
}

/// A stretch: rows of one group between clock events, with their
/// policy times from 0, and how it ended.
struct Stretch {
    rows: Vec<usize>,
    tau: Vec<f64>,
    /// How it ended, and the policy time of the end, if it ended.
    end: Option<(End, f64)>,
}

/// The clock from stretch row `from` to stretch row `to`, as the core
/// decides an edge: the raw clocks' exact difference on a temporal
/// clock, the raw clocks' on a number clock (task 159, W3), else the
/// policy times' (review R2, W1).
fn gap_bf(rows: &[Row], s: &Stretch, from: usize, to: usize) -> Gap {
    match (rows[s.rows[from]].clock, rows[s.rows[to]].clock) {
        (Some(ClockValue::Ns(x)), Some(ClockValue::Ns(y))) => Gap::Ns(y - x),
        (Some(ClockValue::F64(x)), Some(ClockValue::F64(y))) => Gap::Secs(y - x),
        (Some(ClockValue::I64(x)), Some(ClockValue::I64(y))) => {
            Gap::Int(i128::from(y) - i128::from(x))
        }
        _ => Gap::Secs(s.tau[to] - s.tau[from]),
    }
}

fn brute(
    kernels: &[KernelDef],
    ops: &[OpDef],
    cfg: ClockCfg,
    rows: &[Row],
) -> Result<Table, Refusal> {
    struct G {
        clock: ClockState,
        tau: f64,
        origin: Option<ClockValue>,
        stretch: usize,
        restart: bool,
        last_raw: Option<ClockValue>,
    }
    // With groups a session is each group's (review R2, W3).
    let grouped = rows.iter().any(|r| r.group.is_some());
    let mut shared = ClockState::new();
    let mut groups: HashMap<Option<String>, G, rheon_core::FixedState> = HashMap::default();
    let mut stretches: Vec<Stretch> = Vec::new();
    let mut live: Vec<bool> = Vec::new();
    let (mut stretch_of, mut tau_of) = (Vec::new(), Vec::new());
    for (i, r) in rows.iter().enumerate() {
        let seq = i as u64;
        let mut sc = shared.clone();
        let adv = sc.advance(&cfg, r.clock, if grouped { None } else { r.session }, true);
        refuse_backwards(&adv, seq, shared.last_clock(), r.clock)?;
        // A stream restart starts every group's clock over (task 159, W1).
        let mut gc = if adv.reset {
            ClockState::new()
        } else {
            groups
                .get(&r.group)
                .map_or_else(ClockState::new, |g| g.clock.clone())
        };
        let prev = gc.last_clock();
        let own = gc.advance(&cfg, r.clock, r.session, true);
        refuse_backwards(&own, seq, prev, r.clock)?;
        shared = sc;
        let end = |g: &mut G, how: End, stretches: &mut Vec<Stretch>, live: &mut Vec<bool>| {
            if live[g.stretch] {
                live[g.stretch] = false;
                // A cut stretch ends at the last row it saw (R1, C4).
                stretches[g.stretch].end = Some((how, g.tau));
            }
            g.restart = true;
        };
        if adv.reset || adv.session_changed || adv.capped {
            let how = if adv.reset { End::Discard } else { End::Cut };
            for g in groups.values_mut() {
                end(g, how, &mut stretches, &mut live);
                if adv.reset {
                    g.clock = ClockState::new();
                }
            }
        }
        if let (Some(now), true) = (r.clock, cfg.gap_cap.is_finite()) {
            for g in groups.values_mut() {
                if let Some(last) = g.last_raw
                    && elapsed_bf(now, last) > cfg.gap_cap
                {
                    end(g, End::Cut, &mut stretches, &mut live);
                }
            }
        }
        let g = groups.entry(r.group.clone()).or_insert_with(|| G {
            clock: ClockState::new(),
            tau: 0.0,
            origin: None,
            stretch: usize::MAX,
            restart: true,
            last_raw: None,
        });
        if own.reset {
            end(g, End::Discard, &mut stretches, &mut live);
        } else if own.session_changed || own.capped {
            end(g, End::Cut, &mut stretches, &mut live);
        }
        g.clock = gc;
        if g.restart {
            g.tau = 0.0;
            g.origin = r.clock;
            g.restart = false;
            g.stretch = stretches.len();
            stretches.push(Stretch {
                rows: Vec::new(),
                tau: Vec::new(),
                end: None,
            });
            live.push(true);
        } else if let (Some(now), Some(origin)) = (r.clock, g.origin) {
            g.tau = elapsed(now, origin);
        } else {
            g.tau += own.d_clock;
            if g.origin.is_none() {
                g.origin = r.clock;
            }
        }
        stretches[g.stretch].rows.push(i);
        stretches[g.stretch].tau.push(g.tau);
        stretch_of.push(g.stretch);
        tau_of.push(g.tau);
        g.last_raw = r.clock;
    }
    let n_out = ops.len();
    let mut t = Table {
        values: vec![vec![f64::NAN; rows.len()]; n_out],
        drop: vec![false; rows.len()],
    };
    let value = |r: usize, input: usize| {
        let v = rows[r].values[input];
        if usable(v) { v } else { f64::NAN }
    };
    for s in &stretches {
        let n = s.rows.len();
        for (o, op) in ops.iter().enumerate() {
            let k = &kernels[op.kernel];
            for a in 0..n {
                let (r, ta) = (s.rows[a], s.tau[a]);
                let stamp = rows[r].clock;
                let (members, complete, span): (Vec<usize>, bool, f64) = match k.direction {
                    Direction::Backward => {
                        let w = k.window_size;
                        // Rows at the stamp, per `closed`; the lower
                        // edge per `closed`.
                        let members: Vec<usize> = (0..n)
                            .filter(|&b| {
                                let at_stamp = rows[s.rows[b]].clock == stamp && stamp.is_some();
                                let before = b < a && !at_stamp
                                    || (at_stamp && b != a && stamp.is_some())
                                    || (b == a);
                                let near_ok = if k.closed.near(Direction::Backward) {
                                    (b <= a) || at_stamp
                                } else {
                                    b < a && !at_stamp
                                };
                                let far_ok = match w {
                                    None => true,
                                    Some(_) => {
                                        let age = gap_bf(rows, s, b, a).cmp_window(k);
                                        if k.closed.far(Direction::Backward) {
                                            age != Ordering::Greater
                                        } else {
                                            age == Ordering::Less
                                        }
                                    }
                                };
                                let _ = before;
                                near_ok && far_ok
                            })
                            .collect();
                        (
                            members,
                            w.is_none_or(|w| ta >= w),
                            w.map_or(ta, |w| w.min(ta)),
                        )
                    }
                    Direction::Forward => {
                        let w = k.window_size.unwrap();
                        // A set of timestamps: under "left" and "both"
                        // every row at the row's stamp, itself included
                        // (R1, C3); under "right" and "none" none at it.
                        let later: Vec<usize> = (0..n)
                            .filter(|&b| {
                                let d = gap_bf(rows, s, a, b);
                                let (near, far) = (d.cmp_zero(), d.cmp_window(k));
                                let near_ok = if k.closed.near(Direction::Forward) {
                                    near != Ordering::Less
                                } else {
                                    near == Ordering::Greater
                                };
                                let far_ok = if k.closed.far(Direction::Forward) {
                                    far != Ordering::Greater
                                } else {
                                    far == Ordering::Less
                                };
                                near_ok && far_ok
                            })
                            .collect();
                        // Whole once every timestamp of the stretch the
                        // window can hold is in, read from the stretch
                        // itself rather than from a closing rule (task
                        // 160, PC5). A stretch a break or a reset ended
                        // has every row it will have: the window is
                        // whole when the stretch reaches its far edge, so
                        // that its span lies inside the stretch (task
                        // 173, PC2, for a reset), and otherwise cut short
                        // by a break, or discarded by a reset. A stretch
                        // still open at the end of the input has shown
                        // its window whole only by a row stamped past
                        // every timestamp the window holds: past the far
                        // edge where the window holds it ("right",
                        // "both"), at it or past where it does not; a
                        // window it has not shown whole is unresolved,
                        // and null -- except that, under "right" and
                        // "both", a window whose far edge is the last
                        // row of the input holds every timestamp it
                        // can, as at a cut, and is whole.
                        let reaches_edge =
                            gap_bf(rows, s, a, n - 1).cmp_window(k) != Ordering::Less;
                        let shown_whole = (a + 1..n).any(|b| {
                            let d = gap_bf(rows, s, a, b).cmp_window(k);
                            if k.closed.far(Direction::Forward) {
                                d == Ordering::Greater
                            } else {
                                d != Ordering::Less
                            }
                        });
                        match s.end {
                            Some((End::Cut | End::Discard, _)) if reaches_edge => (later, true, w),
                            Some((End::Cut, end_tau)) => {
                                if op.partial == Partial::Drop {
                                    t.drop[r] = true;
                                }
                                if op.partial != Partial::Keep || !rows[r].accept {
                                    continue;
                                }
                                (later, false, (ta + w).min(end_tau) - ta)
                            }
                            _ if shown_whole
                                || (k.closed.far(Direction::Forward)
                                    && gap_bf(rows, s, a, n - 1).cmp_window(k)
                                        == Ordering::Equal) =>
                            {
                                (later, true, w)
                            }
                            _ => continue,
                        }
                    }
                };
                if k.direction == Direction::Backward {
                    if !complete && op.partial == Partial::Drop {
                        t.drop[r] = true;
                    }
                    if !complete && op.partial != Partial::Keep {
                        continue;
                    }
                } else if !rows[r].accept {
                    continue;
                }
                // The value, from the definition.
                let (mut sum, mut mass, mut count) = (0.0, 0.0, 0.0);
                let mut held: Vec<(f64, f64)> = Vec::new();
                let mut weighed: Vec<(f64, f64)> = Vec::new();
                for &b in &members {
                    let x = value(s.rows[b], op.input);
                    if x.is_nan() {
                        continue;
                    }
                    count += 1.0;
                    let tb = s.tau[b];
                    match op.stat {
                        Stat::Mean | Stat::Var | Stat::Std => {
                            let (lo, hi) = match k.direction {
                                Direction::Backward => {
                                    // Held from the operator's last valued row.
                                    let prev = (0..b)
                                        .rev()
                                        .find(|&c| !value(s.rows[c], op.input).is_nan());
                                    let start = match prev {
                                        Some(c) => s.tau[c],
                                        None => k
                                            .window_size
                                            .map_or(f64::NEG_INFINITY, |w| s.tau[0] - w),
                                    };
                                    let edge = k.window_size.map_or(f64::NEG_INFINITY, |w| ta - w);
                                    // A row exactly a window old holds its
                                    // value up to the edge and no further,
                                    // decided from the raw clocks as its
                                    // membership is (task 430).
                                    let on_edge = k.window_size.is_some()
                                        && gap_bf(rows, s, b, a).cmp_window(k) == Ordering::Equal;
                                    if on_edge {
                                        (tb, tb)
                                    } else {
                                        (start.max(edge), tb)
                                    }
                                }
                                Direction::Forward => {
                                    // Held until the operator's next valued row.
                                    let next =
                                        (b + 1..n).find(|&c| !value(s.rows[c], op.input).is_nan());
                                    let end = match next {
                                        Some(c) => s.tau[c],
                                        None => s.end.map_or(f64::INFINITY, |(_, e)| e),
                                    };
                                    let far = ta + k.window_size.unwrap();
                                    // As backward: a row exactly a window
                                    // ahead holds nothing inside it.
                                    if gap_bf(rows, s, a, b).cmp_window(k) == Ordering::Equal {
                                        (tb, tb)
                                    } else {
                                        (
                                            tb,
                                            end.min(far)
                                                .min(s.end.map_or(f64::INFINITY, |(_, e)| e)),
                                        )
                                    }
                                }
                            };
                            let m = mass_bf(k, ta, lo, hi);
                            sum += m * x;
                            mass += m;
                            held.push((m, x));
                        }
                        Stat::Moments => unreachable!(),
                        Stat::Sum
                        | Stat::Rate
                        | Stat::Quantile
                        | Stat::Max
                        | Stat::Min
                        | Stat::PointMean
                        | Stat::ExactQuantile
                        | Stat::ExactRank => {
                            let d = (tb - ta).abs();
                            let f = if k.half_life.is_infinite() {
                                1.0
                            } else {
                                (-(d / k.half_life)).exp2()
                            };
                            sum += f * x;
                            mass += f;
                            weighed.push((x, f));
                        }
                    }
                }
                if count < f64::from(op.min_samples) {
                    continue;
                }
                t.values[o][r] = match op.stat {
                    Stat::Mean => {
                        if mass > 0.0 {
                            sum / mass
                        } else {
                            f64::NAN
                        }
                    }
                    Stat::Sum => sum,
                    Stat::Var | Stat::Std => var_bf(&held, op.bias, op.stat == Stat::Std),
                    Stat::Quantile => quantile_bf(&weighed, op.quantile.unwrap()),
                    Stat::Moments => unreachable!(),
                    // Folded oldest first, ties to the earliest looking
                    // back and to the latest looking ahead.
                    Stat::Max | Stat::Min => weighed.iter().fold(f64::NAN, |a, &(x, _)| {
                        extrema::pick(
                            op.stat == Stat::Max,
                            k.direction == Direction::Forward,
                            a,
                            x,
                        )
                    }),
                    Stat::PointMean => {
                        if mass > 0.0 {
                            sum / mass
                        } else {
                            f64::NAN
                        }
                    }
                    Stat::ExactQuantile => {
                        let mut ord = OrdSide::default();
                        for &(x, _) in &weighed {
                            ord.insert(x);
                        }
                        ord.quantile(
                            op.quantile.unwrap(),
                            op.interpolation.unwrap_or(Interp::Nearest),
                            None,
                        )
                    }
                    Stat::ExactRank => {
                        let mut ord = OrdSide::default();
                        for &(x, _) in &weighed {
                            ord.insert(x);
                        }
                        ord.rank(
                            value(s.rows[a], op.input),
                            op.rank_method.unwrap_or(RankMethod::Average),
                            None,
                        )
                    }
                    Stat::Rate => {
                        let m = mass_bf(k, 0.0, 0.0, span).abs();
                        let m = if k.direction == Direction::Backward {
                            mass_bf(k, span, 0.0, span)
                        } else {
                            m
                        };
                        if m > 0.0 { sum / m } else { f64::NAN }
                    }
                };
            }
        }
    }
    let _ = (&stretch_of, &tau_of);
    Ok(t)
}

/// The quantile at `p` of `(value, weight)` pairs as the sketch defines
/// it: each value in its bucket (`window_quantile::Key`), the buckets in
/// value order, and the representative of the first at which the weight
/// reaches `p` of the total. NaN with no weight.
fn quantile_bf(weighed: &[(f64, f64)], p: f64) -> f64 {
    use crate::window_quantile::Key;
    let rank = |k: Key| match k {
        Key::Neg(b) => (0, -i64::from(b)),
        Key::Zero => (1, 0),
        Key::Pos(b) => (2, i64::from(b)),
    };
    let mut keyed: Vec<(Key, f64)> = weighed.iter().map(|&(x, w)| (Key::of(x), w)).collect();
    keyed.sort_by_key(|&(k, _)| rank(k));
    let total: f64 = keyed.iter().map(|&(_, w)| w).sum();
    if total.is_nan() || total <= 0.0 {
        return f64::NAN;
    }
    let mut buckets: Vec<(Key, f64)> = Vec::new();
    for (k, w) in keyed {
        match buckets.last_mut() {
            Some((last, sum)) if *last == k => *sum += w,
            _ => buckets.push((k, w)),
        }
    }
    let mut cum = 0.0;
    for &(k, w) in &buckets {
        cum += w;
        if w > 0.0 && cum >= p * total {
            return k.value();
        }
    }
    buckets.last().map_or(f64::NAN, |&(k, _)| k.value())
}

/// The variance of `held`'s `(weight, value)` pairs from the definition,
/// in two passes: the weighted mean, then the weighted mean square
/// deviation from it; unless `bias`, times `V1² / (V1² − V2)` (null
/// where one weight is all of it); its root for a standard deviation.
fn var_bf(held: &[(f64, f64)], bias: bool, std: bool) -> f64 {
    let v1: f64 = held.iter().map(|&(m, _)| m).sum();
    if v1 <= 0.0 {
        return f64::NAN;
    }
    let mu = held.iter().map(|&(m, x)| m * x).sum::<f64>() / v1;
    let mut var = held
        .iter()
        .map(|&(m, x)| m * (x - mu) * (x - mu))
        .sum::<f64>()
        / v1;
    if !bias {
        // `V1² − V2 = 2 Σ_{i<j} m_i m_j`, summed as the products of each
        // weight with the ones before it: every term at or above 0.
        let (mut before, mut pairs) = (0.0, 0.0);
        for &(m, _) in held {
            pairs += m * before;
            before += m;
        }
        if pairs <= 0.0 {
            return f64::NAN;
        }
        var *= v1 * v1 / (2.0 * pairs);
    }
    if std { var.sqrt() } else { var }
}

/// A test window's nanoseconds, as the frame reads them off a duration.
fn ns_of(w: f64) -> i64 {
    (w * 1e9).round() as i64
}

fn cfg(cap: f64, session_gap: Option<SessionGap>, restart: Option<f64>) -> ClockCfg {
    ClockCfg {
        gap_cap: cap,
        on_clock_reset: if restart.is_some() {
            OnClockReset::ResetState
        } else {
            OnClockReset::Error
        },
        session_gap,
        min_backwards_jump: restart.unwrap_or(0.0),
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() * n as f64) as usize
    }
}

/// A stream with gaps, repeated stamps, session changes, missing
/// values and several groups.
fn stream(seed: u64, n: usize, groups: usize, temporal: bool, sessions: bool) -> Vec<Row> {
    let mut rng = Lcg(seed);
    let mut t = 0.0f64;
    let mut rows = Vec::with_capacity(n);
    let mut session = 0u64;
    for i in 0..n {
        let step = match rng.below(10) {
            0 => 0.0,
            1 | 2 => 0.5,
            3..=7 => 1.0 + rng.next(),
            8 => 4.0,
            _ => 12.0,
        };
        t += step;
        if sessions && i > 0 && rng.below(40) == 0 {
            session += 1;
        }
        let g = rng.below(groups);
        let value = |rng: &mut Lcg| {
            if rng.below(6) == 0 {
                f64::NAN
            } else {
                rng.next() * 10.0 - 3.0
            }
        };
        let v0 = value(&mut rng);
        let v1 = value(&mut rng);
        rows.push(Row {
            group: if groups > 1 {
                Some(format!("g{g}"))
            } else {
                None
            },
            clock: Some(if temporal {
                ClockValue::Ns((t * 1e9).round() as i64)
            } else {
                ClockValue::F64(t)
            }),
            session: sessions.then_some(session),
            values: vec![v0, v1],
            accept: rng.below(8) != 0,
        });
    }
    rows
}

/// [`stream`] on a half-unit grid: every step a multiple of 0.5, so a
/// row lands exactly a window after another, and exactly on a window's
/// far edge as the last row before a break, often (task 160, PC2;
/// `stream`'s random steps land there only where its exact ones happen
/// to add up). `repeats` lets a step of 0 repeat a stamp.
fn grid_stream(seed: u64, n: usize, groups: usize, sessions: bool, repeats: bool) -> Vec<Row> {
    let mut rows = stream(seed, n, groups, false, sessions);
    let mut rng = Lcg(seed.wrapping_mul(31).wrapping_add(7));
    let mut t = 0.0f64;
    for r in &mut rows {
        t += match rng.below(9) {
            0 if repeats => 0.0,
            0 | 1 => 0.5,
            2 | 3 => 1.0,
            4 => 1.5,
            5 | 6 => 2.0,
            7 => 3.0,
            _ => 12.0,
        };
        r.clock = Some(ClockValue::F64(t));
    }
    rows
}

fn all_kernels() -> (Vec<KernelDef>, Vec<OpDef>) {
    let mut kernels = Vec::new();
    let mut ops = Vec::new();
    for (direction, half_life, window, closed) in [
        (Direction::Backward, 3.0, None, Closed::Right),
        (Direction::Backward, 3.0, Some(5.0), Closed::Right),
        (Direction::Backward, 2.0, Some(4.0), Closed::Left),
        (Direction::Backward, f64::INFINITY, Some(6.0), Closed::Both),
        (Direction::Backward, 5.0, Some(7.0), Closed::Neither),
        (Direction::Forward, 3.0, Some(5.0), Closed::Right),
        (Direction::Forward, 2.0, Some(4.0), Closed::Left),
        (Direction::Forward, f64::INFINITY, Some(6.0), Closed::Both),
        (Direction::Forward, 5.0, Some(3.0), Closed::Neither),
    ] {
        let ki = kernels.len();
        kernels.push(KernelDef {
            direction,
            half_life,
            window_size: window,
            window_ns: window.map(ns_of),
            closed,
            since: None,
        });
        for (stat, partial, min) in [
            (Stat::Mean, Partial::Keep, 1),
            (Stat::Sum, Partial::Null, 2),
            (Stat::Rate, Partial::Keep, 1),
            (Stat::Mean, Partial::Drop, 1),
        ] {
            // A drop operator on every kernel would drop most rows.
            if partial == Partial::Drop && ki != 5 {
                continue;
            }
            ops.push(OpDef {
                kernel: ki,
                stat,
                input: ops.len() % 2,
                min_samples: min,
                partial,
                bias: false,
                quantile: None,
                max_state_bytes: None,
                interpolation: None,
                rank_method: None,
            });
        }
        // A quantile on every kernel, low and high, so its walk runs up
        // from the smallest value and down from the largest (task 272
        // (d)); `min_samples` of 2 looking back, where a window of 3 ahead
        // holds two rows on no row of one integer stream.
        let min = if direction == Direction::Backward {
            2
        } else {
            1
        };
        for (partial, min, level) in [(Partial::Keep, 1, 0.3), (Partial::Null, min, 0.85)] {
            ops.push(OpDef {
                kernel: ki,
                stat: Stat::Quantile,
                input: ops.len() % 2,
                min_samples: min,
                partial,
                bias: false,
                quantile: Some(level),
                max_state_bytes: None,
                interpolation: None,
                rank_method: None,
            });
        }
        // A variance and a standard deviation on every backward kernel,
        // corrected and not (task 212).
        if direction == Direction::Backward {
            for (stat, partial, min, bias) in [
                (Stat::Var, Partial::Keep, 1, false),
                (Stat::Std, Partial::Null, 2, true),
            ] {
                ops.push(OpDef {
                    kernel: ki,
                    stat,
                    input: ops.len() % 2,
                    min_samples: min,
                    partial,
                    bias,
                    quantile: None,
                    max_state_bytes: None,
                    interpolation: None,
                    rank_method: None,
                });
            }
        }
        // A maximum on the two even kernels, beside every other
        // operator, a variance's wide queue among them (task 561).
        if half_life.is_infinite() {
            ops.push(extremum(ki, Stat::Max, Partial::Keep, 1));
        }
    }
    // The extrema's own kernels, every `closed` each way and the running
    // form; a maximum and a minimum on each.
    for (direction, window, closed) in [
        (Direction::Backward, None, Closed::Right),
        (Direction::Backward, Some(5.0), Closed::Right),
        (Direction::Backward, Some(4.0), Closed::Left),
        (Direction::Backward, Some(7.0), Closed::Neither),
        (Direction::Forward, Some(5.0), Closed::Right),
        (Direction::Forward, Some(4.0), Closed::Left),
        (Direction::Forward, Some(3.0), Closed::Neither),
    ] {
        let ki = kernels.len();
        kernels.push(KernelDef {
            direction,
            half_life: f64::INFINITY,
            window_size: window,
            window_ns: window.map(ns_of),
            closed,
            since: None,
        });
        let partial = if window.is_some() {
            Partial::Null
        } else {
            Partial::Keep
        };
        ops.push(extremum(ki, Stat::Max, Partial::Keep, 1));
        ops.push(extremum(ki, Stat::Min, partial, 2));
    }
    (kernels, ops)
}

fn extremum(kernel: usize, stat: Stat, partial: Partial, min_samples: u32) -> OpDef {
    OpDef {
        kernel,
        stat,
        input: usize::from(stat == Stat::Min),
        min_samples,
        partial,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    }
}

/// Every operator on every kernel against the definition, on streams
/// with gaps, repeated stamps, sessions, missing values and groups.
#[test]
fn every_operator_matches_its_definition() {
    let (kernels, ops) = all_kernels();
    // The last four on the half-unit grid, where windows end exactly on
    // the last row before a break (task 160, PC2).
    for (seed, groups, temporal, sessions, grid) in [
        (1, 1, false, false, false),
        (2, 1, true, false, false),
        (3, 3, false, true, false),
        (4, 2, true, true, false),
        (5, 4, true, false, false),
        (6, 1, false, false, true),
        (7, 1, false, true, true),
        (8, 3, false, true, true),
        (9, 2, false, false, true),
    ] {
        let rows = if grid {
            grid_stream(seed, 400, groups, sessions, true)
        } else {
            stream(seed, 400, groups, temporal, sessions)
        };
        let c = cfg(8.0, sessions.then_some(SessionGap::Gap(2.0)), None);
        assert_matches_brute(&kernels, &ops, c, &rows, seed);
    }
}

/// Task 200: on an integer clock of epoch nanoseconds near 1.79e18,
/// where a double resolves 256, the core decides every edge, gap and
/// silence from the integers: it matches the brute force, whose edges
/// are the integers' differences, and gives what the same stream
/// shifted to 0 gives as a float clock, every value of which a double
/// holds -- to the bit. The streams are the half-unit grid doubled, so
/// rows land exactly a window apart and exactly at the cap, often.
#[test]
fn an_integer_clock_decides_its_edges_in_integers() {
    const T0: i64 = 1_790_000_000_000_000_000;
    let (kernels, ops) = all_kernels();
    for (seed, groups, sessions) in [(21u64, 1usize, false), (22, 3, true), (23, 2, false)] {
        let grid = grid_stream(seed, 400, groups, sessions, true);
        let at = |r: &Row| match r.clock {
            Some(ClockValue::F64(t)) => (2.0 * t) as i64,
            _ => unreachable!("the grid is a float clock"),
        };
        let int: Vec<Row> = grid
            .iter()
            .map(|r| Row {
                clock: Some(ClockValue::I64(T0 + at(r))),
                ..r.clone()
            })
            .collect();
        let shifted: Vec<Row> = grid
            .iter()
            .map(|r| Row {
                clock: Some(ClockValue::F64(at(r) as f64)),
                ..r.clone()
            })
            .collect();
        let c = cfg(8.0, sessions.then_some(SessionGap::Gap(2.0)), None);
        assert_matches_brute(&kernels, &ops, c, &int, seed);
        let (a, b) = (
            run(&kernels, &ops, c, &int, 1).unwrap(),
            run(&kernels, &ops, c, &shifted, 1).unwrap(),
        );
        if let Err((_, _, e)) = a.close_to(&b, 0.0) {
            panic!("seed {seed}: {e}");
        }
    }
}

/// The core against the brute force on `rows`, to 1e-9, the rows around
/// the first difference printed.
fn assert_matches_brute(
    kernels: &[KernelDef],
    ops: &[OpDef],
    c: ClockCfg,
    rows: &[Row],
    seed: u64,
) {
    let want = brute(kernels, ops, c, rows).unwrap();
    let got = run(kernels, ops, c, rows, 1).unwrap();
    // A match where a variance is never valued holds nothing of it (a
    // sparse forward sum is valued on no row of one integer stream).
    // Nor where a quantile is never valued (task 272 (d)).
    for (o, col) in want.values.iter().enumerate() {
        let valued = col.iter().filter(|v| !v.is_nan()).count();
        assert!(
            valued > 0 || !(ops[o].stat.is_var() || ops[o].stat == Stat::Quantile),
            "seed {seed}: output {o} valued on {valued} rows"
        );
    }
    if let Err((o, i, e)) = got.close_to(&want, 1e-9) {
        for (r, row) in rows
            .iter()
            .enumerate()
            .take(i + 5)
            .skip(i.saturating_sub(10))
        {
            let (g, w) = if o < got.values.len() {
                (got.values[o][r], want.values[o][r])
            } else {
                (f64::NAN, f64::NAN)
            };
            eprintln!(
                "row {r}: clock {:?} x {:?} accept {} got {g} want {w}",
                row.clock, row.values, row.accept
            );
        }
        panic!("seed {seed}: {e}");
    }
}

/// Every operator on every kernel against the definition across resets
/// (task 173, PC2): the half-unit grid's breaks made steps back past
/// `restart_after_step_back` or new sessions under `session_gap =
/// "reset"`, alone and with groups, so windows end exactly on the last
/// row before a reset. Whole there, as before a cut; the rest of the
/// windows a reset meets discarded, null under every `partial` and
/// never dropped.
#[test]
fn every_operator_matches_its_definition_across_resets() {
    let (kernels, ops) = all_kernels();
    for (seed, groups) in [(21, 1), (22, 3), (23, 1), (24, 2)] {
        let (rows, c) = if seed < 23 {
            (
                step_back_stream(seed, 400, groups, true),
                cfg(8.0, None, Some(0.0)),
            )
        } else {
            (
                session_reset_stream(seed, 400, groups, true),
                cfg(8.0, Some(SessionGap::Reset), None),
            )
        };
        assert_matches_brute(&kernels, &ops, c, &rows, seed);
    }
}

/// Feeding a stream in chunks of any size gives the same bits.
/// Review R1, C7: the core refused this only through the formula layer;
/// a direct caller got a NaN mean.
#[test]
fn a_mean_with_no_decay_needs_a_window() {
    let err = Windows::new(
        vec![KernelDef {
            direction: Direction::Backward,
            half_life: f64::INFINITY,
            window_size: None,
            window_ns: None,
            closed: Closed::Right,
            since: None,
        }],
        vec![OpDef {
            kernel: 0,
            stat: Stat::Mean,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
            quantile: None,
            max_state_bytes: None,
            interpolation: None,
            rank_method: None,
        }],
        ClockCfg::default(),
    )
    .expect_err("refused");
    assert!(err.contains("needs a window_size"), "{err}");
}

#[test]
fn chunking_changes_no_bit() {
    let (kernels, ops) = all_kernels();
    let rows = stream(7, 500, 3, true, true);
    let c = cfg(8.0, Some(SessionGap::Gap(2.0)), None);
    let one = run(&kernels, &ops, c, &rows, usize::MAX).unwrap();
    for every in [1, 2, 3, 7, 50, 499] {
        let t = run(&kernels, &ops, c, &rows, every).unwrap();
        assert!(t.same_bits(&one), "every {every}");
    }
}

/// A test row's number clock.
fn clock_of(r: &Row) -> f64 {
    match r.clock {
        Some(ClockValue::F64(x)) => x,
        _ => unreachable!("a number clock"),
    }
}

/// [`grid_stream`]'s clock with each of its steps past the cap turned
/// into a step back of the same size, which `restart_after_step_back`
/// restarts on: the same stretches, each ended by a reset rather than a
/// cut (task 173, PC2).
fn step_back_stream(seed: u64, n: usize, groups: usize, repeats: bool) -> Vec<Row> {
    let mut rows = grid_stream(seed, n, groups, false, repeats);
    let (mut prev, mut t) = (0.0, 0.0);
    for r in &mut rows {
        let step = clock_of(r) - prev;
        prev = clock_of(r);
        t += if step == 12.0 { -12.0 } else { step };
        r.clock = Some(ClockValue::F64(t));
    }
    rows
}

/// [`grid_stream`]'s clock with each of its steps past the cap turned
/// into a step of 1 and a new session, which `session_gap = "reset"`
/// restarts on: the same stretches again, each ended by the other kind
/// of reset (task 173, PC2).
fn session_reset_stream(seed: u64, n: usize, groups: usize, repeats: bool) -> Vec<Row> {
    let mut rows = grid_stream(seed, n, groups, false, repeats);
    let (mut prev, mut t, mut session) = (0.0, 0.0, 0u64);
    for r in &mut rows {
        let step = clock_of(r) - prev;
        prev = clock_of(r);
        if step == 12.0 {
            session += 1;
            t += 1.0;
        } else {
            t += step;
        }
        r.clock = Some(ClockValue::F64(t));
        r.session = Some(session);
    }
    rows
}

/// The time-reversal identity on `rows`, under `c`: row `i`'s forward
/// window is the mirror's backward window at the mirrored row, where
/// the mirror is the stream with its times negated and its rows
/// reversed. `ends(j)` says a break follows row `j`, which ends its
/// stretch: the mirror has the same break between the same two rows (a
/// gap, a step back of the same size, the same new session). The last
/// row is put after one, so the input ends on a stretch of one row and
/// no forward window is left open by the end of the input where its
/// mirror is whole (task 160, PC2).
fn assert_mirror(rows: &[Row], c: ClockCfg, ends: impl Fn(&[Row], usize) -> bool) {
    let n = rows.len();
    let t: Vec<f64> = rows.iter().map(clock_of).collect();
    // The rows whose window of 5 ends exactly on the last row before a
    // break: the case the mirror catches.
    let on_edge = (0..n)
        .filter(|&i| {
            let end = (i..n).find(|&j| j + 1 == n || ends(rows, j)).unwrap();
            end + 1 < n && t[end] - t[i] == 5.0
        })
        .count();
    assert!(on_edge > 5, "{on_edge} rows end on the edge");
    let last = t[n - 1];
    let mut mirror: Vec<Row> = rows.iter().rev().cloned().collect();
    for r in &mut mirror {
        r.clock = Some(ClockValue::F64(last - clock_of(r)));
    }
    // The mirror: times negated and reversed, a backward kernel closed
    // on the far side -- "left" mirrors "right", "both" mirrors itself.
    for (closed, mirrored) in [(Closed::Right, Closed::Left), (Closed::Both, Closed::Both)] {
        let k = KernelDef {
            direction: Direction::Forward,
            half_life: 3.0,
            window_size: Some(5.0),
            window_ns: Some(ns_of(5.0)),
            closed,
            since: None,
        };
        let ops: Vec<OpDef> = [
            Stat::Mean,
            Stat::Sum,
            Stat::Rate,
            Stat::Quantile,
            Stat::Quantile,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, stat)| OpDef {
            kernel: 0,
            stat,
            input: 0,
            min_samples: 1,
            partial: Partial::Null,
            bias: false,
            quantile: (stat == Stat::Quantile).then_some(if i == 3 { 0.2 } else { 0.7 }),
            max_state_bytes: None,
            interpolation: None,
            rank_method: None,
        })
        .collect();
        let fwd = run(std::slice::from_ref(&k), &ops, c, rows, 1).unwrap();
        let kb = KernelDef {
            direction: Direction::Backward,
            closed: mirrored,
            ..k
        };
        let back = run(&[kb], &ops, c, &mirror, 1).unwrap();
        // Row i's forward window is the mirror's backward window at the
        // mirrored row i: whole where it is whole, cut short or discarded
        // where it does not reach its far edge (null both), and the same
        // number where it is whole.
        for o in 0..ops.len() {
            for (i, ti) in t.iter().enumerate() {
                let (a, b) = (fwd.values[o][i], back.values[o][n - 1 - i]);
                assert_eq!(
                    a.is_nan(),
                    b.is_nan(),
                    "{closed:?} op {o} row {i} at {ti}: {a} vs {b}"
                );
                if !a.is_nan() {
                    assert!(
                        (a - b).abs() <= 1e-9 * (1.0 + a.abs()),
                        "{closed:?} op {o} row {i}: {a} vs {b}"
                    );
                }
            }
            assert!(fwd.values[o].iter().filter(|v| !v.is_nan()).count() > n / 3);
        }
    }
}

/// A forward window is a backward one over the reversed stream, less the
/// row itself: the mirror of each kernel on the mirror of the stream.
#[test]
fn a_forward_window_is_a_backward_one_over_the_reversed_stream() {
    // Distinct stamps on a half-unit grid, so the row itself is the only
    // exclusion and a row lands exactly on a window's far edge, broken
    // by gaps past the cap.
    let mut rows = grid_stream(11, 300, 1, false, false);
    rows.iter_mut().for_each(|r| r.accept = true);
    let mut last_row = rows.last().unwrap().clone();
    last_row.clock = Some(ClockValue::F64(clock_of(&last_row) + 12.0));
    rows.push(last_row);
    assert_mirror(&rows, cfg(8.0, None, None), |rows, j| {
        clock_of(&rows[j + 1]) - clock_of(&rows[j]) > 8.0
    });
}

/// The identity across resets (task 173, PC2): a reset discards every
/// window still open across it, but one whose far edge is the last row
/// before it is whole, as at a cut -- every row it covers has arrived,
/// and its mirror, a backward window reaching exactly back to its
/// stretch's first row, is complete. Before, a reset discarded it, and
/// the forward side was null where the mirror had its number.
#[test]
fn a_forward_window_is_a_backward_one_over_the_reversed_stream_across_resets() {
    // A step back past `restart_after_step_back` (0: every step back).
    let mut rows = step_back_stream(11, 300, 1, false);
    rows.iter_mut().for_each(|r| r.accept = true);
    let mut last_row = rows.last().unwrap().clone();
    last_row.clock = Some(ClockValue::F64(clock_of(&last_row) - 12.0));
    rows.push(last_row);
    assert_mirror(&rows, cfg(8.0, None, Some(0.0)), |rows, j| {
        clock_of(&rows[j + 1]) < clock_of(&rows[j])
    });
    // A new session under `session_gap = "reset"`.
    let mut rows = session_reset_stream(11, 300, 1, false);
    rows.iter_mut().for_each(|r| r.accept = true);
    let mut last_row = rows.last().unwrap().clone();
    last_row.clock = Some(ClockValue::F64(clock_of(&last_row) + 1.0));
    last_row.session = last_row.session.map(|s| s + 1);
    rows.push(last_row);
    assert_mirror(&rows, cfg(8.0, Some(SessionGap::Reset), None), |rows, j| {
        rows[j + 1].session != rows[j].session
    });
}

/// A stream of many groups, interleaved, `per` rows at each stamp on
/// average (1 to `2 * per - 1`), each row's group drawn afresh: rows
/// pending in many groups at once at a repeated stamp.
fn many_groups_stream(seed: u64, n: usize, groups: usize, per: usize) -> Vec<Row> {
    let mut rows = stream(seed, n, groups, false, false);
    let mut rng = Lcg(seed.wrapping_mul(17).wrapping_add(3));
    let (mut t, mut left) = (0.0, 0);
    for r in &mut rows {
        if left == 0 {
            t += 1.0;
            left = 1 + rng.below(2 * per - 1);
        }
        left -= 1;
        r.group = Some(format!("g{}", rng.below(groups)));
        r.clock = Some(ClockValue::F64(t));
    }
    rows
}

/// One backward sum under `closed`, with no window and no cap: every
/// group stays on the silent list for the whole stream.
fn one_sum(closed: Closed) -> (Vec<KernelDef>, Vec<OpDef>) {
    let k = KernelDef {
        direction: Direction::Backward,
        half_life: 50.0,
        window_size: None,
        window_ns: None,
        closed,
        since: None,
    };
    let op = OpDef {
        kernel: 0,
        stat: Stat::Sum,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    (vec![k], vec![op])
}

/// A row resolves only the groups with rows pending (task 325): the
/// walk from the silent list's head stepped over every group ever seen
/// on every row, each already resolved, so the rows' cost grew with the
/// groups, not the work -- 80 µs a row at 10,000 groups. Counted, not
/// timed: over 2,000 interleaved groups, `"left"` and `"none"` pend no
/// row and resolve no group as rows arrive; `"right"` and `"both"`
/// resolve each group once for each stamp it had rows at, at the next
/// stamp, and never one with nothing pending. The head walk made about
/// 2,000 calls a row here, nearly all idle.
#[test]
fn a_row_resolves_only_the_groups_with_rows_pending() {
    let (n, groups) = (6_000, 2_000);
    let rows = many_groups_stream(41, n, groups, 4);
    let last = rows.last().and_then(|r| r.clock);
    // Each group's stamps before the last: one resolution each.
    let mut due = std::collections::HashSet::new();
    for r in rows.iter().filter(|r| r.clock != last) {
        let Some(ClockValue::F64(t)) = r.clock else {
            unreachable!("a number clock")
        };
        due.insert((r.group.clone(), t.to_bits()));
    }
    assert!(due.len() > n / 2, "{} group stamps", due.len());
    for closed in [Closed::Left, Closed::Neither, Closed::Right, Closed::Both] {
        let (kernels, ops) = one_sum(closed);
        let mut core = Windows::new(kernels, ops, cfg(f64::INFINITY, None, None)).unwrap();
        core.set_grouped(true);
        FLUSHES.with(|c| c.set((0, 0)));
        for r in &rows {
            push(&mut core, r).unwrap();
        }
        let (calls, idle) = FLUSHES.with(std::cell::Cell::get);
        assert!(
            core.groups.len() > groups * 9 / 10,
            "{closed:?}: groups seen"
        );
        let want = if closed.near(Direction::Backward) {
            due.len() as u64
        } else {
            0
        };
        assert_eq!((calls, idle), (want, 0), "{closed:?}: calls, idle");
    }
}

/// A state saved while many groups have rows pending at a repeated
/// stamp resumes to the bit (task 325): the walk reads the pending
/// groups off the silent list's tail, which the state carries.
#[test]
fn a_state_saved_with_many_groups_pending_resumes_to_the_bit() {
    let (kernels, ops) = all_kernels();
    let rows = many_groups_stream(43, 600, 60, 6);
    let c = cfg(f64::INFINITY, None, None);
    let one = run(&kernels, &ops, c, &rows, 1).unwrap();
    let mut most = 0;
    for at in [1, 5, 97, 211, 300, 451, 599] {
        let mut core = Windows::new(kernels.clone(), ops.clone(), c).unwrap();
        core.set_grouped(true);
        let (mut t, mut next) = (Table::default(), 0);
        for r in &rows[..at] {
            push(&mut core, r).unwrap();
        }
        most = most.max(core.groups.iter().filter(|g| !g.pending.is_empty()).count());
        t.append(core.drain(), &mut next);
        let bytes = rmp_serde::to_vec_named(&core).unwrap();
        let mut core: Windows = rmp_serde::from_slice(&bytes).unwrap();
        core.check().unwrap();
        for r in &rows[at..] {
            push(&mut core, r).unwrap();
        }
        t.append(core.finish(), &mut next);
        assert!(t.same_bits(&one), "resumed at {at}");
    }
    assert!(most >= 4, "at most {most} groups pending at a save");
}

/// Every row at a repeated stamp gets the same backward window under
/// "right", later rows at the stamp included; under "left" the stamp's
/// rows are outside and the window is known as the stamp arrives.
#[test]
fn rows_at_one_stamp_share_a_backward_window() {
    let rows: Vec<Row> = [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (2.0, 4.0),
        (2.0, 5.0),
        (3.0, 6.0),
        (5.0, 7.0),
    ]
    .iter()
    .map(|&(t, x)| Row {
        group: None,
        clock: Some(ClockValue::F64(t)),
        session: None,
        values: vec![x],
        accept: true,
    })
    .collect();
    let c = cfg(100.0, None, None);
    let sums = |closed: Closed| {
        let k = KernelDef {
            direction: Direction::Backward,
            half_life: f64::INFINITY,
            window_size: Some(2.0),
            window_ns: Some(ns_of(2.0)),
            closed,
            since: None,
        };
        let op = OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
            quantile: None,
            max_state_bytes: None,
            interpolation: None,
            rank_method: None,
        };
        run(&[k], &[op], c, &rows, 1).unwrap().values[0].clone()
    };
    // Polars' rolling_sum_by, window 2, on these stamps (measured
    // 2026-10-02, docs/PLAN.md task 143).
    assert_eq!(
        sums(Closed::Right),
        vec![1.0, 3.0, 14.0, 14.0, 14.0, 18.0, 7.0]
    );
    assert_eq!(
        sums(Closed::Both),
        vec![1.0, 3.0, 15.0, 15.0, 15.0, 20.0, 13.0]
    );
    let left = sums(Closed::Left);
    assert!(left[0].is_nan() && left[1..] == [1.0, 3.0, 3.0, 3.0, 14.0, 6.0]);
    let none = sums(Closed::Neither);
    assert!(none[0].is_nan() && none[1..6] == [1.0, 2.0, 2.0, 2.0, 12.0] && none[6].is_nan());
}

/// The backward time-weighted mean is `ewm_mean_by`'s recursion with
/// `a_1 = 1`, written out, and a repeated stamp moves it not at all.
#[test]
fn the_backward_mean_is_the_recursion() {
    let stamps = [0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 5.0, 5.5, 9.0];
    let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let rows: Vec<Row> = stamps
        .iter()
        .zip(xs)
        .map(|(&t, x)| Row {
            group: None,
            clock: Some(ClockValue::F64(t)),
            session: None,
            values: vec![x],
            accept: true,
        })
        .collect();
    let h = 1.0;
    let k = KernelDef {
        direction: Direction::Backward,
        half_life: h,
        window_size: None,
        window_ns: None,
        closed: Closed::Right,
        since: None,
    };
    let op = OpDef {
        kernel: 0,
        stat: Stat::Mean,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let got = run(&[k], &[op], cfg(100.0, None, None), &rows, 1)
        .unwrap()
        .values[0]
        .clone();
    let mut y = xs[0];
    let mut want = vec![y];
    for i in 1..xs.len() {
        let a = 1.0 - (-(stamps[i] - stamps[i - 1]) / h).exp2();
        y = a * xs[i] + (1.0 - a) * y;
        want.push(y);
    }
    // Rows at one stamp share the window: the stamp's last recursion
    // value, which the zero-mass rows did not move.
    let want: Vec<f64> = (0..xs.len())
        .map(|i| {
            let last = (i..xs.len())
                .take_while(|&j| stamps[j] == stamps[i])
                .last()
                .unwrap();
            want[last]
        })
        .collect();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-12, "row {i}: {g} vs {w}");
    }
}

/// One row's variance by Polars' own `ewm_var(adjust=False)` algorithm
/// (pandas' `ewmcov`), written out: the old weight scaled by `1 − α`
/// and the new row's by `α`, the weights renormalized to a sum of 1, the
/// mean and the co-moment moved together, and unless `bias` the
/// co-moment times `sum_wt² / (sum_wt² − sum_wt2)`.
fn polars_ewm_var(xs: &[f64], h: f64, bias: bool) -> Vec<f64> {
    let alpha = 1.0 - (-1.0 / h).exp2();
    let (mut mean, mut cov) = (xs[0], 0.0);
    let (mut old_wt, mut sum_wt, mut sum_wt2) = (1.0f64, 1.0f64, 1.0f64);
    let mut out = vec![if bias { 0.0 } else { f64::NAN }];
    for &x in &xs[1..] {
        sum_wt *= 1.0 - alpha;
        sum_wt2 *= (1.0 - alpha) * (1.0 - alpha);
        old_wt *= 1.0 - alpha;
        let old_mean = mean;
        let wt_sum = old_wt + alpha;
        mean = (old_wt * old_mean + alpha * x) / wt_sum;
        cov = (old_wt * (cov + (old_mean - mean) * (old_mean - mean))
            + alpha * (x - mean) * (x - mean))
            / wt_sum;
        sum_wt += alpha;
        sum_wt2 += alpha * alpha;
        old_wt += alpha;
        sum_wt /= old_wt;
        sum_wt2 /= old_wt * old_wt;
        old_wt = 1.0;
        out.push(if bias {
            cov
        } else {
            let num = sum_wt * sum_wt;
            let den = num - sum_wt2;
            if den > 0.0 { num / den * cov } else { f64::NAN }
        });
    }
    out
}

/// On a clock that steps by 1, with `half_life` in rows, the variance is
/// Polars' `ewm_var(adjust=False)`, whose algorithm is written out above:
/// a stretch's first value held from before it weighs `(1 − α)^(n−1)`,
/// as `adjust=False`'s first row does (task 212). The standard deviation
/// is its root. The Python suite holds the same to Polars itself.
#[test]
fn the_variance_on_a_unit_clock_is_polars_adjust_false() {
    let mut rng = Lcg(77);
    let xs: Vec<f64> = (0..300).map(|_| rng.next() * 10.0 - 3.0).collect();
    let rows: Vec<Row> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| Row {
            group: None,
            clock: Some(ClockValue::F64(i as f64)),
            session: None,
            values: vec![x],
            accept: true,
        })
        .collect();
    for (h, bias) in [(1.0, false), (6.5, true), (6.5, false), (40.0, false)] {
        let k = KernelDef {
            direction: Direction::Backward,
            half_life: h,
            window_size: None,
            window_ns: None,
            closed: Closed::Right,
            since: None,
        };
        let op = |stat| OpDef {
            kernel: 0,
            stat,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias,
            quantile: None,
            max_state_bytes: None,
            interpolation: None,
            rank_method: None,
        };
        let got = run(
            &[k],
            &[op(Stat::Var), op(Stat::Std)],
            cfg(100.0, None, None),
            &rows,
            1,
        )
        .unwrap();
        let want = polars_ewm_var(&xs, h, bias);
        for (i, w) in want.iter().enumerate() {
            let (v, s) = (got.values[0][i], got.values[1][i]);
            assert_eq!(
                v.is_nan(),
                w.is_nan(),
                "h {h} bias {bias} row {i}: {v} vs {w}"
            );
            if !w.is_nan() {
                assert!(
                    (v - w).abs() <= 1e-12 * w.abs().max(1.0),
                    "h {h} row {i}: {v} vs {w}"
                );
                assert_eq!(s.to_bits(), v.sqrt().to_bits(), "row {i}");
            }
        }
    }
}

/// Every window holds one value however many rows it weighs, so its
/// variance is 0 exactly, at any level and on every kernel: the merge
/// of two segments moves the mean by their means' difference, 0, and
/// adds nothing to `M2` (task 212). A mean square less a squared mean
/// would cancel to rounding noise at a level.
#[test]
fn a_constant_input_has_a_variance_of_exactly_zero() {
    let (kernels, ops) = all_kernels();
    let var: Vec<usize> = (0..ops.len()).filter(|&o| ops[o].stat.is_var()).collect();
    for level in [0.1, 1e8 + 0.1, -3e12, 7.0] {
        let rows: Vec<Row> = stream(5, 400, 2, true, true)
            .into_iter()
            .map(|mut r| {
                for v in &mut r.values {
                    if !v.is_nan() {
                        *v = level;
                    }
                }
                r
            })
            .collect();
        let c = cfg(8.0, Some(SessionGap::Gap(2.0)), None);
        let t = run(&kernels, &ops, c, &rows, 1).unwrap();
        for &o in &var {
            let valued: Vec<f64> = t.values[o]
                .iter()
                .copied()
                .filter(|v| !v.is_nan())
                .collect();
            assert!(
                valued.len() > 20,
                "level {level} output {o}: {}",
                valued.len()
            );
            assert!(
                valued.iter().all(|&v| v == 0.0),
                "level {level} output {o}: {:?}",
                valued.iter().find(|&&v| v != 0.0)
            );
        }
    }
}

/// A variance's mean is a compensated pair (`rheon_core::comp`): a
/// value held row after row is approached as exact arithmetic approaches
/// it. At 1e8 a plain mean stalls a few rounding steps short of a new
/// level, and the variance about it settles on that gap's square, about
/// 1e-15; the definition decays as `p (1 − p)`, `p = 2^(−T/h)` the weight
/// left on the old level `T` units after its last row. The core's
/// discount is an `exp2`, so the bound is relative, not bitwise.
#[test]
fn a_variance_at_a_level_decays_with_its_history() {
    let (a, n_a, n, h) = (1e8, 50usize, 500usize, 5.0);
    let rows: Vec<Row> = (0..n)
        .map(|i| Row {
            group: None,
            clock: Some(ClockValue::F64(i as f64)),
            session: None,
            values: vec![if i < n_a { a } else { a + 1.0 }],
            accept: true,
        })
        .collect();
    let k = KernelDef {
        direction: Direction::Backward,
        half_life: h,
        window_size: None,
        window_ns: None,
        closed: Closed::Right,
        since: None,
    };
    let op = OpDef {
        kernel: 0,
        stat: Stat::Var,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: true,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let v = run(&[k], &[op], cfg(100.0, None, None), &rows, 1)
        .unwrap()
        .values[0]
        .clone();
    for t in [n_a + 20, n_a + 100, n_a + 200, n - 1] {
        let p = (-((t - (n_a - 1)) as f64) / h).exp2();
        let want = p * (1.0 - p);
        assert!(
            (v[t] - want).abs() <= 1e-9 * want,
            "row {t}: {} vs {want}",
            v[t]
        );
    }
}

/// A variance looks back only, and needs a window without decay, as a
/// mean does: refused when the core is built, and in a core read back
/// from a damaged state.
#[test]
fn a_variance_looks_back_only() {
    let k = |direction| KernelDef {
        direction,
        half_life: 2.0,
        window_size: Some(4.0),
        window_ns: None,
        closed: Closed::Right,
        since: None,
    };
    let op = |kernel, stat| OpDef {
        kernel,
        stat,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let err = Windows::new(
        vec![k(Direction::Forward)],
        vec![op(0, Stat::Std)],
        ClockCfg::default(),
    )
    .expect_err("refused");
    assert!(
        err.contains("a variance reads only the rows up to each row"),
        "{err}"
    );
    let err = Windows::new(
        vec![KernelDef {
            half_life: f64::INFINITY,
            window_size: None,
            ..k(Direction::Backward)
        }],
        vec![op(0, Stat::Var)],
        ClockCfg::default(),
    )
    .expect_err("refused");
    assert!(err.contains("needs a window_size"), "{err}");
    let mut core = Windows::new(
        vec![k(Direction::Backward), k(Direction::Forward)],
        vec![op(0, Stat::Var), op(1, Stat::Sum)],
        ClockCfg::default(),
    )
    .unwrap();
    core.check().unwrap();
    Arc::make_mut(&mut core.ops)[0].kernel = 1;
    let err = core.check().expect_err("refused");
    assert!(
        err.contains("a variance reads only the rows up to each row"),
        "{err}"
    );
}

/// A reset discards the windows open across it, whatever `partial`
/// says; a cut keeps, nulls or drops them as it says. A window whose far
/// edge is the last row before either is whole.
#[test]
fn a_reset_discards_and_a_cut_follows_partial() {
    let k = KernelDef {
        direction: Direction::Forward,
        half_life: 2.0,
        window_size: Some(5.0),
        window_ns: Some(ns_of(5.0)),
        closed: Closed::Right,
        since: None,
    };
    let ops = vec![
        OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
            quantile: None,
            max_state_bytes: None,
            interpolation: None,
            rank_method: None,
        },
        OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Null,
            bias: false,
            quantile: None,
            max_state_bytes: None,
            interpolation: None,
            rank_method: None,
        },
        OpDef {
            kernel: 0,
            stat: Stat::Sum,
            input: 0,
            min_samples: 1,
            partial: Partial::Drop,
            bias: false,
            quantile: None,
            max_state_bytes: None,
            interpolation: None,
            rank_method: None,
        },
    ];
    let mk = |ts: &[f64]| -> Vec<Row> {
        ts.iter()
            .map(|&t| Row {
                group: None,
                clock: Some(ClockValue::F64(t)),
                session: None,
                values: vec![1.0],
                accept: true,
            })
            .collect()
    };
    // A gap past the cap at row 3 cuts rows 0..3's windows.
    let cut = run(
        std::slice::from_ref(&k),
        &ops,
        cfg(4.0, None, None),
        &mk(&[0.0, 1.0, 2.0, 20.0, 21.0]),
        1,
    )
    .unwrap();
    assert!(cut.values[0][0] > 0.0 && cut.values[1][0].is_nan() && cut.drop[0]);
    assert!(!cut.drop[3]);
    // A step back restarts: discarded, null, never dropped.
    let reset = run(
        std::slice::from_ref(&k),
        &ops,
        cfg(4.0, None, Some(0.0)),
        &mk(&[0.0, 1.0, 2.0, 1.5, 2.5]),
        1,
    )
    .unwrap();
    assert!(reset.values[0][0].is_nan() && reset.values[1][0].is_nan() && !reset.drop[0]);
    // Task 173, PC2: row 0's window (0, 5] ends exactly on the last row
    // before the step back, so every row it covers has arrived: whole,
    // under every `partial`, as before a cut. Its sum from the
    // definition, `Σ 2^(−d/2)` over the rows 1, 2 and 5 later. The
    // windows that reach past that row are discarded, never dropped.
    let reset = run(
        &[k],
        &ops,
        cfg(100.0, None, Some(0.0)),
        &mk(&[0.0, 1.0, 2.0, 5.0, 3.0]),
        1,
    )
    .unwrap();
    let whole = 2f64.powf(-0.5) + 2f64.powf(-1.0) + 2f64.powf(-2.5);
    // `(1.25 k + 2)·2⁻⁵²` for a libm `k` ulps off (docs/PLAN.md task
    // 273): each side takes `2^(−1/2)` (`k` ulps of `2⁻⁵³`) and
    // `2^(−5/2)` (`k` of `2⁻⁵⁵`) from its libm, `exp2` here and `powf`
    // above, `2⁻¹` exactly, and rounds two sums near 1.4 (½ ulp each).
    // Held at `8·2⁻⁵²`, a libm up to 4.8 ulps off: measured 0 with
    // Apple's, 3 with Apple's moved by up to 2 ulps and 5 by up to 4,
    // where `1e-15` (4.5), the bound before, failed.
    for (o, values) in reset.values.iter().enumerate().take(3) {
        assert!(
            (values[0] - whole).abs() <= 8.0 * f64::EPSILON,
            "op {o}: {} vs {whole}",
            values[0]
        );
        assert!(values[1..].iter().all(|v| v.is_nan()), "op {o}");
    }
    assert!(reset.drop.iter().all(|d| !d), "{:?}", reset.drop);
}

/// `min_samples` nulls a window with fewer rows carrying a value.
#[test]
fn min_samples_counts_rows_with_a_value() {
    let rows: Vec<Row> = [1.0, f64::NAN, 2.0, 3.0]
        .iter()
        .enumerate()
        .map(|(i, &x)| Row {
            group: None,
            clock: Some(ClockValue::F64(i as f64)),
            session: None,
            values: vec![x],
            accept: true,
        })
        .collect();
    let k = KernelDef {
        direction: Direction::Backward,
        half_life: f64::INFINITY,
        window_size: Some(3.0),
        window_ns: Some(ns_of(3.0)),
        closed: Closed::Right,
        since: None,
    };
    let op = OpDef {
        kernel: 0,
        stat: Stat::Sum,
        input: 0,
        min_samples: 2,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let got = run(&[k], &[op], cfg(100.0, None, None), &rows, 1)
        .unwrap()
        .values[0]
        .clone();
    assert!(got[0].is_nan() && got[1].is_nan() && got[2] == 3.0 && got[3] == 5.0);
}

/// The state survives msgpack at every row and goes on identically.
#[test]
fn a_saved_state_resumes_at_any_row() {
    let (kernels, ops) = all_kernels();
    // Under `session_gap = "reset"` too, whose saved form no other
    // state carries (task 252): it must load as itself, not as no
    // session gap.
    for (rows, c) in [
        (
            stream(9, 120, 2, true, true),
            cfg(8.0, Some(SessionGap::Gap(2.0)), None),
        ),
        (
            session_reset_stream(10, 120, 2, true),
            cfg(8.0, Some(SessionGap::Reset), None),
        ),
    ] {
        let one = run(&kernels, &ops, c, &rows, 1).unwrap();
        for at in [1, 2, 17, 60, 119] {
            let mut core = Windows::new(kernels.clone(), ops.clone(), c).unwrap();
            core.set_grouped(true);
            let (mut t, mut next) = (Table::default(), 0);
            for r in &rows[..at] {
                push(&mut core, r).unwrap();
            }
            t.append(core.drain(), &mut next);
            let bytes = rmp_serde::to_vec_named(&core).unwrap();
            let mut core: Windows = rmp_serde::from_slice(&bytes).unwrap();
            core.check().unwrap();
            assert_eq!(core.clock_cfg(), &c, "resumed at {at}");
            for r in &rows[at..] {
                push(&mut core, r).unwrap();
            }
            t.append(core.finish(), &mut next);
            assert!(t.same_bits(&one), "{c:?}: resumed at {at}");
        }
    }
}

/// A step back is refused by name with the step, and nothing moves.
#[test]
fn a_step_back_is_refused_and_changes_nothing() {
    let k = KernelDef {
        direction: Direction::Backward,
        half_life: 2.0,
        window_size: None,
        window_ns: None,
        closed: Closed::Right,
        since: None,
    };
    let op = OpDef {
        kernel: 0,
        stat: Stat::Sum,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let mut core = Windows::new(vec![k], vec![op], cfg(10.0, None, None)).unwrap();
    let row = |t: f64| Row {
        group: None,
        clock: Some(ClockValue::F64(t)),
        session: None,
        values: vec![1.0],
        accept: true,
    };
    push(&mut core, &row(0.0)).unwrap();
    push(&mut core, &row(2.0)).unwrap();
    let before = rmp_serde::to_vec_named(&core).unwrap();
    let err = push(&mut core, &row(1.0)).unwrap_err();
    assert!(matches!(err, Refusal::Backwards { seq: 2, back, .. } if back == 1.0));
    assert_eq!(rmp_serde::to_vec_named(&core).unwrap(), before);
}

/// Review R2, W1: an edge was decided from two rounded policy times, so
/// a row exactly one window from another landed on either side of it
/// (0.4 - 0.1 is not 0.3 in a double); it is decided from the two rows'
/// clocks, exact in nanoseconds, where Polars puts it, at any age.
#[test]
fn an_edge_is_decided_from_the_two_rows_clocks() {
    let op = OpDef {
        kernel: 0,
        stat: Stat::Sum,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let same = |got: &[f64], want: &[f64]| {
        got.len() == want.len()
            && got
                .iter()
                .zip(want)
                .all(|(x, y)| x.is_nan() && y.is_nan() || x == y)
    };
    const EPOCH_2024_MS: i64 = 1_704_067_200_000;
    for base in [0, EPOCH_2024_MS] {
        for (direction, closed, ms, want) in [
            // Backward "left": the row 300 ms older is at the far edge, in.
            (
                Direction::Backward,
                Closed::Left,
                [0, 100, 400, 800],
                [f64::NAN, 1.0, 1.0, f64::NAN],
            ),
            // Backward "right": the row 300 ms older is at the far edge, out.
            (
                Direction::Backward,
                Closed::Right,
                [0, 400, 700, 1100],
                [1.0, 1.0, 1.0, 1.0],
            ),
            // Forward "right": the row 300 ms later is at the far edge, in. A
            // forward window closes only at a stamp past its far edge, so the
            // fourth row is there to close row 1's; row 2's closes empty, and
            // row 3's is unresolved at the end.
            (
                Direction::Forward,
                Closed::Right,
                [0, 100, 400, 800],
                [1.0, 1.0, f64::NAN, f64::NAN],
            ),
        ] {
            let k = KernelDef {
                direction,
                half_life: f64::INFINITY,
                window_size: Some(0.3),
                window_ns: Some(ns_of(0.3)),
                closed,
                since: None,
            };
            let rows: Vec<Row> = ms
                .iter()
                .map(|&m| Row {
                    group: None,
                    clock: Some(ClockValue::Ns((base + m) * 1_000_000)),
                    session: None,
                    values: vec![1.0],
                    accept: true,
                })
                .collect();
            let c = || cfg(100.0, None, None);
            let got = run(
                std::slice::from_ref(&k),
                std::slice::from_ref(&op),
                c(),
                &rows,
                1,
            )
            .unwrap();
            assert!(
                same(&got.values[0], &want),
                "{direction:?} {closed:?} from {base}: {:?}",
                got.values[0]
            );
            let bf = brute(
                std::slice::from_ref(&k),
                std::slice::from_ref(&op),
                c(),
                &rows,
            )
            .unwrap();
            assert!(got.same_bits(&bf), "the brute force decides the same edges");
        }
    }
}

/// Review R2, W3: the stream's clock took every row's session, so groups
/// with sessions of their own saw a change at every row, and every
/// group's windows started over at every row. With groups a session is
/// each group's; without, the stream is the one group and its session
/// change still ends the windows.
#[test]
fn with_groups_a_session_is_each_groups() {
    let k = KernelDef {
        direction: Direction::Backward,
        half_life: f64::INFINITY,
        window_size: None,
        window_ns: None,
        closed: Closed::Right,
        since: None,
    };
    let op = OpDef {
        kernel: 0,
        stat: Stat::Sum,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let row = |t: f64, g: Option<&str>, s: u64, x: f64| Row {
        group: g.map(str::to_string),
        clock: Some(ClockValue::F64(t)),
        session: Some(s),
        values: vec![x],
        accept: true,
    };
    let sums = |rows: &[Row]| {
        run(
            std::slice::from_ref(&k),
            std::slice::from_ref(&op),
            cfg(10.0, Some(SessionGap::Gap(1.0)), None),
            rows,
            1,
        )
        .unwrap()
        .values[0]
            .clone()
    };
    let (a, b) = (Some("a"), Some("b"));
    let rows = [
        row(0.0, a, 1, 1.0),
        row(1.0, b, 2, 10.0),
        row(2.0, a, 1, 3.0),
        row(3.0, b, 2, 30.0),
    ];
    assert_eq!(sums(&rows), [1.0, 10.0, 4.0, 40.0]);
    // A group's own session change ends its windows, and no other's.
    let rows = [
        row(0.0, a, 1, 1.0),
        row(1.0, b, 2, 10.0),
        row(2.0, a, 3, 3.0),
        row(3.0, b, 2, 30.0),
    ];
    assert_eq!(sums(&rows), [1.0, 10.0, 3.0, 40.0]);
    // One group: the stream's session, a change at rows 1, 2 and 3.
    let rows = [
        row(0.0, None, 1, 1.0),
        row(1.0, None, 2, 10.0),
        row(2.0, None, 3, 3.0),
        row(3.0, None, 2, 30.0),
    ];
    assert_eq!(sums(&rows), [1.0, 10.0, 3.0, 30.0]);
}

/// The seconds path, for a window given as a number on a temporal
/// clock (no `window_ns`): the gap converts with the same operations a
/// duration's text does, so every edge falls where the integer path
/// puts it.
#[test]
fn a_numeric_window_on_a_temporal_clock_decides_the_same_edges() {
    let op = OpDef {
        kernel: 0,
        stat: Stat::Sum,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let rows: Vec<Row> = [0, 100, 400, 800]
        .iter()
        .map(|&m| Row {
            group: None,
            clock: Some(ClockValue::Ns(m * 1_000_000)),
            session: None,
            values: vec![1.0],
            accept: true,
        })
        .collect();
    for (direction, closed) in [
        (Direction::Backward, Closed::Left),
        (Direction::Backward, Closed::Right),
        (Direction::Forward, Closed::Right),
        (Direction::Forward, Closed::Left),
    ] {
        let integer = KernelDef {
            direction,
            half_life: f64::INFINITY,
            window_size: Some(0.3),
            window_ns: Some(300_000_000),
            closed,
            since: None,
        };
        let seconds = KernelDef {
            window_ns: None,
            ..integer.clone()
        };
        let ops = std::slice::from_ref(&op);
        let a = run(
            std::slice::from_ref(&integer),
            ops,
            cfg(100.0, None, None),
            &rows,
            1,
        )
        .unwrap();
        let b = run(
            std::slice::from_ref(&seconds),
            ops,
            cfg(100.0, None, None),
            &rows,
            1,
        )
        .unwrap();
        assert!(
            a.same_bits(&b),
            "{direction:?} {closed:?}: {a:?} against {b:?}"
        );
        assert!(
            a.values[0].iter().filter(|v| !v.is_nan()).count() >= 2,
            "edges exercised"
        );
    }
}

/// Review R4, A1, recorded and not changed: under groups the stream's
/// clock cuts a group silent past `gap_cap` when another group's row
/// shows the silence, before the group's next row can say its session
/// changed; a `session_gap = "reset"` then discards only what is still
/// open. The same rows without a group column discard, since the one
/// clock sees the gap and the session change at one row, and the reset
/// comes first.
#[test]
fn a_silent_groups_reset_after_a_capped_gap_is_a_cut() {
    let k = KernelDef {
        direction: Direction::Forward,
        half_life: f64::INFINITY,
        window_size: Some(10.0),
        window_ns: None,
        closed: Closed::Right,
        since: None,
    };
    let op = OpDef {
        kernel: 0,
        stat: Stat::Sum,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let row = |t: f64, g: Option<&str>, s: u64, x: f64| Row {
        group: g.map(str::to_string),
        clock: Some(ClockValue::F64(t)),
        session: Some(s),
        values: vec![x],
        accept: true,
    };
    let (a, b) = (Some("a"), Some("b"));
    let rows = [
        row(0.0, a, 1, 1.0),
        row(1.0, b, 1, 10.0),
        row(2.0, b, 1, 20.0),
        row(4.0, a, 1, 1.0),
        row(8.0, a, 1, 1.0),
        row(9.0, b, 2, 30.0),
    ];
    let c = || cfg(5.0, Some(SessionGap::Reset), None);
    let grouped = run(
        std::slice::from_ref(&k),
        std::slice::from_ref(&op),
        c(),
        &rows,
        1,
    )
    .unwrap();
    assert_eq!(
        grouped.values[0][1], 20.0,
        "cut at t = 8: the window over what b saw"
    );
    let alone: Vec<Row> = rows
        .iter()
        .filter(|r| r.group.as_deref() == b)
        .map(|r| Row {
            group: None,
            ..r.clone()
        })
        .collect();
    let one = run(
        std::slice::from_ref(&k),
        std::slice::from_ref(&op),
        c(),
        &alone,
        1,
    )
    .unwrap();
    assert!(
        one.values[0][0].is_nan(),
        "discarded at t = 9: {:?}",
        one.values[0]
    );
}

/// Review R4, A4: a kernel's two forms of its window agree, or the
/// kernel is refused.
#[test]
fn a_kernels_window_ns_must_be_its_window_size() {
    let op = OpDef {
        kernel: 0,
        stat: Stat::Sum,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let k = KernelDef {
        direction: Direction::Backward,
        half_life: 1.0,
        window_size: Some(0.3),
        window_ns: Some(300_000_001),
        closed: Closed::Right,
        since: None,
    };
    let err = Windows::new(vec![k], vec![op], cfg(10.0, None, None)).expect_err("refused");
    assert!(
        err.contains("window_ns 300000001 is not window_size 0.3"),
        "{err}"
    );
}

/// Task 252: the Rust API's own refusals, which the formula layer's
/// checks keep every `with_windows` call from reaching -- a kernel's
/// half-life, a forward kernel without a window, a window that is not
/// finite and above 0, no operator, an operator naming a kernel there
/// is not or counting no row, and a kernel no operator uses.
#[test]
fn the_core_refuses_what_it_cannot_run_by_name() {
    let kernel = |direction, half_life, window_size| KernelDef {
        direction,
        half_life,
        window_size,
        window_ns: None,
        closed: Closed::Right,
        since: None,
    };
    for h in [0.0, -1.0, f64::NAN] {
        let err = kernel(Direction::Backward, h, Some(2.0))
            .check()
            .unwrap_err();
        assert!(
            err.starts_with("half_life must be above 0, but it is"),
            "{h}: {err}"
        );
    }
    let err = kernel(Direction::Forward, 1.0, None).check().unwrap_err();
    assert!(
        err.starts_with("window_size is not set, but an operator that reads the rows after"),
        "{err}"
    );
    for w in [0.0, -2.0, f64::INFINITY, f64::NAN] {
        for direction in [Direction::Backward, Direction::Forward] {
            let err = kernel(direction, 1.0, Some(w)).check().unwrap_err();
            assert!(
                err.starts_with("window_size must be finite and above 0, but it is"),
                "{w}: {err}"
            );
        }
    }
    kernel(Direction::Backward, 1.0, None).check().unwrap();
    kernel(Direction::Forward, f64::INFINITY, Some(3.0))
        .check()
        .unwrap();
    let op = |kernel: usize, min_samples: u32| OpDef {
        kernel,
        stat: Stat::Sum,
        input: 0,
        min_samples,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
        max_state_bytes: None,
        interpolation: None,
        rank_method: None,
    };
    let back = || kernel(Direction::Backward, 1.0, Some(2.0));
    let refused = |kernels: Vec<KernelDef>, ops: Vec<OpDef>| {
        Windows::new(kernels, ops, cfg(10.0, None, None)).expect_err("refused")
    };
    assert_eq!(
        refused(vec![back()], vec![]),
        defect("kernel 0 has no operator")
    );
    // Empty kernels and ops builds a clock-only core (0 outputs, 0 queued rows).
    let mut clock_only = Windows::new(vec![], vec![], cfg(5.0, None, Some(2.0))).unwrap();
    assert_eq!(clock_only.n_outputs(), 0);
    clock_only.set_grouped(true);
    let g0 = clock_only.group(Some("a"));
    let g1 = clock_only.group(Some("b"));
    for (g, t, s) in [
        (g0, 0.0, 1u64),
        (g1, 1.0, 1),
        (g0, 2.0, 1),
        (g0, 10.0, 1), // gap_cap cut on g0 and silent cut on g1
        (g1, 11.0, 2), // session change on g1
        (g0, 6.0, 1),  // step back > restart_after_step_back (2.0): reset
    ] {
        clock_only
            .push(&RowIn {
                group: g,
                clock: Some(ClockValue::F64(t)),
                session: Some(s),
                values: &[],
                accept: true,
                lists: &[],
            })
            .unwrap();
        clock_only.check().unwrap();
    }
    assert_eq!(clock_only.queued(), 0);
    let out = clock_only.drain();
    assert_eq!(out.rows, 6);
    assert!(out.values.is_empty());
    assert_eq!(clock_only.held(), 0);
    // Step back <= restart_after_step_back is still refused.
    assert!(matches!(
        clock_only.push(&RowIn {
            group: g0,
            clock: Some(ClockValue::F64(5.0)),
            session: Some(1),
            values: &[],
            accept: true,
            lists: &[],
        }),
        Err(Refusal::Backwards { .. })
    ));
    // WIN-9 span overflow past f64::MAX is still refused.
    let mut span_core = Windows::new(vec![], vec![], cfg(1e308, None, None)).unwrap();
    let sg = span_core.group(Some(""));
    span_core
        .push(&RowIn {
            group: sg,
            clock: Some(ClockValue::F64(-1e308)),
            session: None,
            values: &[],
            accept: true,
            lists: &[],
        })
        .unwrap();
    span_core
        .push(&RowIn {
            group: sg,
            clock: Some(ClockValue::F64(0.0)),
            session: None,
            values: &[],
            accept: true,
            lists: &[],
        })
        .unwrap();
    assert!(matches!(
        span_core.push(&RowIn {
            group: sg,
            clock: Some(ClockValue::F64(1e308)),
            session: None,
            values: &[],
            accept: true,
            lists: &[],
        }),
        Err(Refusal::Span { .. })
    ));
    assert_eq!(
        refused(vec![back()], vec![op(3, 1)]),
        defect("operator 0 names kernel 3, which there is not")
    );
    // Only a sum takes 0.
    assert!(
        Windows::new(
            vec![back()],
            vec![op(0, 1), op(0, 0)],
            cfg(10.0, None, None)
        )
        .is_ok()
    );
    let mean = OpDef {
        stat: Stat::Mean,
        ..op(0, 0)
    };
    assert_eq!(
        refused(vec![back()], vec![op(0, 1), mean]),
        defect("min_samples must be at least 1 (only a sum may take 0)")
    );
    assert_eq!(
        refused(vec![back(), back()], vec![op(1, 1)]),
        defect("kernel 0 has no operator")
    );
    let err = refused(
        vec![kernel(Direction::Backward, -1.0, None)],
        vec![op(0, 1)],
    );
    assert!(err.contains("half_life must be above 0"), "{err}");
    // An increment is no statistic of a window, and reads back one row.
    assert_eq!(OpKind::Increment.stat(), None);
    assert_eq!(OpKind::Increment.direction(), Direction::Backward);
    assert_eq!(OpKind::EwmStd.stat(), Some(Stat::Std));
    // The row operators (task 564): no statistic, looking back, each
    // named as it parses; no window operator is one.
    for k in [
        OpKind::Increment,
        OpKind::RunLength,
        OpKind::ClockLag,
        OpKind::ClockDrift,
        OpKind::ClockStep,
    ] {
        assert!(k.is_row() && k.stat().is_none(), "{k:?}");
        assert_eq!(k.direction(), Direction::Backward);
        assert_eq!(OpKind::parse(k.name()), Some(k));
    }
    assert!(!OpKind::EwmSum.is_row() && !OpKind::RewmQuantile.is_row());
}

/// Task 252: what the core reports of itself -- its outputs, one per
/// operator, and the rows its queues hold -- which a run reads to say
/// what its windows cost. A window of 5 over rows a unit apart is read,
/// and evicted to, at the next stamp under `"right"` (equal stamps share
/// a window): after row `t` the queue holds the 5 rows of row `t - 1`'s
/// window `(t - 6, t - 1]` and row `t` itself, 6 however long the
/// stream. The end of input reads the last row's window, evicting to its
/// 5 rows, and keeps them: a state saved there goes on with the next
/// file, whose rows look back into them.
#[test]
fn the_core_counts_its_outputs_and_the_rows_its_windows_hold() {
    let k = KernelDef {
        direction: Direction::Backward,
        half_life: 3.0,
        window_size: Some(5.0),
        window_ns: None,
        closed: Closed::Right,
        since: None,
    };
    let ops: Vec<OpDef> = [Stat::Sum, Stat::Mean, Stat::Var]
        .into_iter()
        .map(|stat| OpDef {
            kernel: 0,
            stat,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
            quantile: None,
            max_state_bytes: None,
            interpolation: None,
            rank_method: None,
        })
        .collect();
    let mut core = Windows::new(vec![k], ops, cfg(10.0, None, None)).unwrap();
    assert_eq!(core.n_outputs(), 3);
    assert_eq!(core.queued(), 0);
    let mut most = 0;
    for i in 0..200 {
        let r = Row {
            group: None,
            clock: Some(ClockValue::F64(f64::from(i))),
            session: None,
            values: vec![f64::from(i % 7)],
            accept: true,
        };
        push(&mut core, &r).unwrap();
        most = most.max(core.queued());
        if i >= 5 {
            assert_eq!(core.queued(), 6, "after row {i}");
        }
        core.drain();
    }
    assert_eq!(most, 6);
    core.finish();
    assert_eq!(core.queued(), 5);
}

/// Task 252: with no cap on a step (`gap_cap` infinite, which only the
/// Rust API can give: a call with a clock column requires a finite
/// one), no silent group is ever cut, and every operator still matches
/// its definition on streams with long gaps, groups and sessions.
#[test]
fn every_operator_matches_its_definition_without_a_cap() {
    let (kernels, ops) = all_kernels();
    for (seed, groups, temporal, sessions) in [
        (31, 3, true, false),
        (32, 2, false, true),
        (33, 1, false, false),
    ] {
        let rows = stream(seed, 300, groups, temporal, sessions);
        let c = cfg(
            f64::INFINITY,
            sessions.then_some(SessionGap::Gap(2.0)),
            None,
        );
        assert_matches_brute(&kernels, &ops, c, &rows, seed);
    }
}

/// Long windows over a stream of `n` rows at steps of 0.5, 1 and 1.5
/// (exact in binary), two groups, a long null run in each input, and a
/// new session two thirds of the way, so each group fills its first
/// window twice: the rows on which a mean's cut reached into `back` and
/// was summed row by row (task 270, M-2).
fn long_window_stream(n: usize) -> Vec<Row> {
    let mut rng = Lcg(270);
    let (mut t, mut rows) = (0.0f64, Vec::with_capacity(n));
    for i in 0..n {
        t += [0.5, 1.0, 1.5][rng.below(3)];
        let session = u64::from(i >= 2 * n / 3);
        let value = |rng: &mut Lcg, null: bool| {
            if null || rng.below(9) == 0 {
                f64::NAN
            } else {
                rng.next() * 10.0 - 3.0
            }
        };
        let v0 = value(&mut rng, (n / 5..n / 5 + 300).contains(&i));
        let v1 = value(&mut rng, i < 150 || (n / 2..n / 2 + 500).contains(&i));
        rows.push(Row {
            group: Some(format!("g{}", rng.below(2))),
            clock: Some(ClockValue::F64(t)),
            session: Some(session),
            values: vec![v0, v1],
            accept: rng.below(10) != 0,
        });
    }
    rows
}

/// Kernels of `window` units, wide enough to hold several hundred rows,
/// at the half-life given: a mean, a variance and a standard deviation
/// on every backward edge, a mean and a sum on every forward one.
fn long_kernels(window: f64, half_life: f64) -> (Vec<KernelDef>, Vec<OpDef>) {
    let (mut kernels, mut ops) = (Vec::new(), Vec::new());
    for direction in [Direction::Backward, Direction::Forward] {
        for closed in [Closed::Right, Closed::Left, Closed::Both, Closed::Neither] {
            let ki = kernels.len();
            kernels.push(KernelDef {
                direction,
                half_life,
                window_size: Some(window),
                window_ns: Some(ns_of(window)),
                closed,
                since: None,
            });
            let stats: &[(Stat, u32, bool)] = if direction == Direction::Backward {
                &[
                    (Stat::Mean, 1, false),
                    (Stat::Var, 2, false),
                    (Stat::Std, 3, true),
                    (Stat::Mean, 40, false),
                ]
            } else {
                &[(Stat::Mean, 1, false), (Stat::Sum, 1, false)]
            };
            for &(stat, min_samples, bias) in stats {
                ops.push(OpDef {
                    kernel: ki,
                    stat,
                    input: ops.len() % 2,
                    min_samples,
                    partial: Partial::Keep,
                    bias,
                    quantile: None,
                    max_state_bytes: None,
                    interpolation: None,
                    rank_method: None,
                });
            }
        }
    }
    (kernels, ops)
}

/// FNV-1a over every output's bits and every drop flag, in order.
fn digest(t: &Table) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |x: u64| {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for col in &t.values {
        for v in col {
            eat(v.to_bits());
        }
    }
    for &d in &t.drop {
        eat(u64::from(d));
    }
    h
}

/// The window operators' every output over long windows, at no decay
/// (no libm on the path, so the bits are every platform's), held to the
/// digest the code before task 270 (M-2) gave on this stream: the fold
/// kept between reads and the search that starts past the rows it has
/// passed over move no bit, at any chunking and across a save. Also the
/// brute force, so the digest is of right numbers.
#[test]
fn long_windows_keep_their_bits() {
    let rows = long_window_stream(3_000);
    let (kernels, ops) = long_kernels(400.0, f64::INFINITY);
    let c = cfg(8.0, Some(SessionGap::Gap(2.0)), None);
    let want = brute(&kernels, &ops, c, &rows).unwrap();
    let one = run(&kernels, &ops, c, &rows, 1).unwrap();
    one.close_to(&want, 1e-9).unwrap();
    // Rows valued on each output: the null runs leave none empty.
    for (o, col) in one.values.iter().enumerate() {
        assert!(col.iter().any(|v| !v.is_nan()), "output {o} never valued");
    }
    assert_eq!(digest(&one), LONG_WINDOWS_DIGEST, "{:#x}", digest(&one));
    for every in [7, 600, usize::MAX] {
        let t = run(&kernels, &ops, c, &rows, every).unwrap();
        assert!(t.same_bits(&one), "every {every}");
    }
    for at in [1, 777, 2_100] {
        let mut core = Windows::new(kernels.clone(), ops.clone(), c).unwrap();
        core.set_grouped(true);
        let (mut t, mut next) = (Table::default(), 0);
        for r in &rows[..at] {
            push(&mut core, r).unwrap();
        }
        t.append(core.drain(), &mut next);
        let bytes = rmp_serde::to_vec_named(&core).unwrap();
        let mut core: Windows = rmp_serde::from_slice(&bytes).unwrap();
        for r in &rows[at..] {
            push(&mut core, r).unwrap();
        }
        t.append(core.finish(), &mut next);
        assert!(t.same_bits(&one), "resumed at {at}");
    }
}

/// The search and the fold as the code before task 270 (M-2) ran them:
/// from the oldest row each read, and `back`'s rows past the cut summed
/// afresh.
fn oldest_by_scan(q: &Queue, pred: impl Fn(&Stack, usize) -> bool) -> Option<(usize, bool, usize)> {
    let n = q.front.len();
    (0..n)
        .map(|p| (p, true, n - 1 - p))
        .chain((0..q.back.len()).map(|i| (n + i, false, i)))
        .find(|&(_, f, i)| pred(q.stack(f), i))
}

fn sum_after_by_rows<const W: bool>(
    q: &Queue,
    k: &KernelDef,
    p: usize,
    j: usize,
) -> Option<(Acc, f64)> {
    let var = W && q.var[j];
    let n = q.front.len();
    if p + 1 < n {
        let mut q = q.clone();
        return q.sum_after_op::<W>(k, p, j);
    }
    let mut sum: Option<(Acc, f64)> = None;
    for i in p + 1 - n..q.back.len() {
        let own = (q.back.own::<W>(i, j), q.back.tau[i]);
        sum = Some(match sum {
            None => own,
            Some((a, a_at)) => then(k, &mut DecayMemo::default(), a, a_at, own.0, own.1, var),
        });
    }
    sum
}

/// What a kernel's read takes for a valued row: a backward one a held
/// value, a forward one a value of the row's own.
fn valued(st: &Stack, i: usize, j: usize, fwd: bool) -> bool {
    match (st.wide, fwd) {
        (true, true) => st.own::<true>(i, j).n > 0.0,
        (true, false) => !st.x::<true>(i, j).is_nan(),
        (false, true) => st.own::<false>(i, j).n > 0.0,
        (false, false) => !st.x::<false>(i, j).is_nan(),
    }
}

fn same_acc(a: Option<(Acc, f64)>, b: Option<(Acc, f64)>) -> bool {
    let bits = |x: Option<(Acc, f64)>| {
        x.map(|(a, at)| [a.s, a.w, a.n, a.lo, a.q, a.e, at].map(f64::to_bits))
    };
    bits(a) == bits(b)
}

/// After every row of a stream through long windows at a finite
/// half-life (where `then` discounts), in every queue, for every
/// operator: the search kept between reads finds the row a scan from
/// the oldest finds, and the fold kept between reads is the bits of
/// the rows summed afresh, at the cut a read takes and at a position
/// drawn at random besides (task 270, M-2).
#[test]
fn the_kept_fold_and_search_are_the_row_by_row_ones() {
    let rows = long_window_stream(2_000);
    let (kernels, ops) = long_kernels(300.0, 30.0);
    let c = cfg(8.0, Some(SessionGap::Gap(2.0)), None);
    let mut core = Windows::new(kernels.clone(), ops, c).unwrap();
    core.set_grouped(true);
    let (mut rng, mut checked, mut into_back) = (Lcg(7), 0usize, 0usize);
    for r in &rows {
        push(&mut core, r).unwrap();
        for g in &mut core.groups {
            for (ki, q) in g.queues.iter_mut().enumerate() {
                let k = &kernels[ki];
                for j in 0..q.n {
                    let fwd = k.direction == Direction::Forward;
                    let pred = |st: &Stack, i: usize| valued(st, i, j, fwd);
                    let want = oldest_by_scan(q, pred);
                    assert_eq!(q.find_oldest(j, pred), want);
                    let len = q.len();
                    let mut ps: Vec<usize> = want.map(|(p, _, _)| p).into_iter().collect();
                    if len > 0 && rng.below(17) == 0 {
                        ps.push(rng.below(len));
                    }
                    for p in ps {
                        into_back += usize::from(p + 1 >= q.front.len() && p + 1 < len);
                        let (a, b) = if q.wide() {
                            (
                                sum_after_by_rows::<true>(q, k, p, j),
                                q.sum_after_op::<true>(k, p, j),
                            )
                        } else {
                            (
                                sum_after_by_rows::<false>(q, k, p, j),
                                q.sum_after_op::<false>(k, p, j),
                            )
                        };
                        assert!(same_acc(a, b), "p {p}: {a:?} against {b:?}");
                        checked += 1;
                    }
                }
            }
        }
    }
    // The cut reached into `back` on most reads: the path this holds.
    assert!(
        into_back > checked / 4,
        "{into_back} of {checked} into back"
    );
}
