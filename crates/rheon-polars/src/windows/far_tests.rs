//! Windows read far from their rows, in half-lives, and at the ends of the
//! doubles (the 2026-11-23 review, PLAN 578d): each output here is held to
//! the definition written out at a scale where nothing underflows, since a
//! mean, a variance or a quantile is a ratio of weights and depends only on
//! the weights' proportions, however small each is seen from the row.
use rheon_core::{ClockCfg, OnClockReset};

use super::*;

const NAN: f64 = f64::NAN;

fn kernel(direction: Direction, half_life: f64, window: Option<f64>, closed: Closed) -> KernelDef {
    KernelDef {
        direction,
        half_life,
        window_size: window,
        window_ns: None,
        closed,
        since: None,
    }
}

fn op(kernel: usize, stat: Stat) -> OpDef {
    OpDef {
        kernel,
        stat,
        input: 0,
        min_samples: 1,
        partial: Partial::Keep,
        bias: false,
        quantile: (stat == Stat::Quantile).then_some(0.5),
        max_state_bytes: None,
    }
}

fn clock(gap_cap: f64) -> ClockCfg {
    ClockCfg {
        gap_cap,
        on_clock_reset: OnClockReset::Error,
        session_gap: None,
        min_backwards_jump: 0.0,
    }
}

/// Each operator's output at each row of `(stamp, value)` rows (NaN for a
/// null), one group, drained after every row and finished at the end; and
/// the core, for a check of its state.
fn outputs(
    kernels: Vec<KernelDef>,
    ops: Vec<OpDef>,
    cfg: ClockCfg,
    rows: &[(f64, f64)],
) -> (Vec<Vec<f64>>, Windows) {
    let n_ops = ops.len();
    let mut core = Windows::new(kernels, ops, cfg).unwrap();
    let mut out = vec![Vec::new(); n_ops];
    for &(t, x) in rows {
        let group = core.group(None);
        core.push(&RowIn {
            group,
            clock: Some(ClockValue::F64(t)),
            session: None,
            values: &[x],
            accept: true,
            lists: &[],
        })
        .unwrap();
        for (o, col) in core.drain().values.into_iter().enumerate() {
            out[o].extend(col);
        }
    }
    for (o, col) in core.finish().values.into_iter().enumerate() {
        out[o].extend(col);
    }
    (out, core)
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-12 * want.abs().max(1.0),
        "{what}: {got} against {want}"
    );
}

/// `∫_a^b λ^(T - s) ds` / `∫_a^b λ^(s - T) ds`, the part of `[a, b]` on
/// the far side of `T` none: the integral itself, by Gauss-Legendre
/// quadrature of the integrand `2^(-|T - s| / h)`, five nodes a panel and
/// no panel longer than half a half-life (the rule's error there is
/// under `1e-17` of the panel). Every term is above 0, so nothing
/// cancels however far `[a, b]` is from `T`, and it shares no closed
/// form with `mass_between`: that took the difference of two masses from
/// `T`, which this function took too, and both were 0 from 54 half-lives
/// (review 2026-11-23, WIN-1). What lies more than 64 half-lives past the
/// interval's near end is under `2^-64` of it and is left out, so an
/// interval with no far end has a sum.
pub(super) fn mass_bf(k: &KernelDef, t: f64, a: f64, b: f64) -> f64 {
    let (a, b) = match k.direction {
        Direction::Backward => (a, b.min(t)),
        Direction::Forward => (a.max(t), b),
    };
    if b <= a {
        return 0.0;
    }
    let h = k.half_life;
    if h.is_infinite() {
        return b - a;
    }
    // From the near end `s0` outward over `len`, `s = s0 ± u`.
    let (s0, sign) = match k.direction {
        Direction::Backward => (b, -1.0),
        Direction::Forward => (a, 1.0),
    };
    let len = (b - a).min(64.0 * h);
    let panels = (len / (0.5 * h)).ceil().max(1.0);
    let width = len / panels;
    // The five-point rule on [-1, 1]: nodes 0 and
    // `±(1/3) sqrt(5 ∓ 2 sqrt(10/7))`, weights `128/225` and
    // `(322 ± 13 sqrt(70)) / 900`.
    let r = (10.0_f64 / 7.0).sqrt();
    let (x1, x2) = ((5.0 - 2.0 * r).sqrt() / 3.0, (5.0 + 2.0 * r).sqrt() / 3.0);
    let s70 = 70.0_f64.sqrt();
    let (w0, w1, w2) = (
        128.0 / 225.0,
        (322.0 + 13.0 * s70) / 900.0,
        (322.0 - 13.0 * s70) / 900.0,
    );
    let rule = [(0.0, w0), (-x1, w1), (x1, w1), (-x2, w2), (x2, w2)];
    let mut total = 0.0;
    // Farthest panel first: the smallest terms are added first.
    for p in (0..panels as usize).rev() {
        let mid = (p as f64 + 0.5) * width;
        let mut panel = 0.0;
        for (x, w) in rule {
            let s = s0 + sign * (mid + 0.5 * width * x);
            panel += w * (-((t - s).abs() / h)).exp2();
        }
        total += 0.5 * width * panel;
    }
    total
}
/// WIN-1, each form of an interval's mass against the integral itself
/// ([`mass_bf`]: quadrature of the decay, no closed form): the plain form,
/// the one a backward window's cut takes through its memo, and the one a
/// forward mean's close takes from its sums' anchor. The intervals lie on
/// either side of the row and across it (only the part on the kernel's
/// side counts), touch it, are empty, and sit up to 900 half-lives from it,
/// where the difference of two masses from the row was 0. With no decay
/// each is the length.
#[test]
fn each_form_of_an_intervals_mass_is_its_integral() {
    let near = |got: f64, want: f64, what: &str| {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs(),
            "{what}: {got:e} against {want:e}"
        );
    };
    for h in [0.3, 1.0, 7.0, f64::INFINITY] {
        for direction in [Direction::Backward, Direction::Forward] {
            let k = kernel(direction, h, Some(1e6), Closed::Right);
            let mut memo = DecayMemo::default();
            // `(a, b)` as distances from the row on the kernel's side.
            for (lo, hi) in [
                (0.0, 0.25),
                (0.5, 2.5),
                (-1.5, 2.0),
                (-3.0, -1.0),
                (2.0, 2.0),
                (10.0, 10.001),
                (53.0, 55.5),
                (200.0, 260.0),
                (900.0 * h.min(1.0), 900.0 * h.min(1.0) + 0.75),
            ] {
                let t = 1000.0;
                let (a, b) = match direction {
                    Direction::Backward => (t - hi, t - lo),
                    Direction::Forward => (t + lo, t + hi),
                };
                let what = format!("h {h} {direction:?} [{a}, {b}] from {t}");
                let want = mass_bf(&k, t, a, b);
                let got = mass_between(&k, t, a, b);
                assert_eq!(want > 0.0, hi > lo.max(0.0), "{what}");
                near(got, want, &what);
                // Through the memo: the same function of the same bits,
                // asked twice.
                for _ in 0..2 {
                    let memoed = mass_between_in(&k, &mut memo, t, a, b);
                    assert_eq!(memoed.to_bits(), got.to_bits(), "{what}, memo");
                }
                if direction == Direction::Forward && a >= t {
                    // A close reads the open row's interval from its sums'
                    // anchor: from the row itself that is the plain form,
                    // and from a time between the row and the interval it
                    // is the mass seen from there.
                    for from in [t, t + 0.5 * lo] {
                        let closed = mass_between_close(&k, &mut memo, t, from, a, b);
                        let seen = if h.is_infinite() { t } else { from };
                        near(
                            closed,
                            mass_bf(&k, seen, a, b),
                            &format!("{what}, close from {from}"),
                        );
                    }
                }
            }
        }
    }
}

/// WIN-1: a backward mean read 54 half-lives after its last value. Its
/// window's mass was `M(far) − M(near)`, which cancels to 0 once both
/// masses round to `h / ln 2`, so a null row read the mean as null and the
/// variance as null, and a row two values old dropped the older value. The
/// windowed mean with a window that holds every row of weight is the
/// unbounded one.
#[test]
fn a_backward_window_read_far_past_its_rows_keeps_their_proportions() {
    let ks = || {
        vec![
            kernel(Direction::Backward, 1.0, None, Closed::Right),
            kernel(Direction::Backward, 1.0, Some(200.0), Closed::Right),
        ]
    };
    let mut var = op(1, Stat::Var);
    var.bias = true;
    let ops = vec![op(0, Stat::Mean), op(1, Stat::Mean), var];
    let (out, _) = outputs(ks(), ops.clone(), clock(1e9), &[(0.0, 42.0), (54.0, NAN)]);
    close(out[0][1], 42.0, "unbounded");
    close(out[1][1], 42.0, "windowed");
    close(out[2][1], 0.0, "windowed variance");
    // 100 held over [−146, 0] and 200 over (0, 1], seen from 54: masses
    // `c (2^-54 − 2^-200)` and `c 2^-54`.
    let (out, _) = outputs(
        ks(),
        ops,
        clock(1e9),
        &[(0.0, 100.0), (1.0, 200.0), (54.0, NAN)],
    );
    let want = (100.0 * (1.0 - 2f64.powi(-146)) + 200.0) / (2.0 - 2f64.powi(-146));
    close(out[0][2], 150.0, "unbounded");
    close(out[1][2], want, "windowed");
}

/// WIN-1, looking ahead: the row at 0 sees 42 held over `[54, 200]`, then
/// 100 over `[54, 55)` and 200 over `[55, 200]`, every mass about `2^-54`
/// of the row's own.
#[test]
fn a_forward_window_read_far_ahead_of_its_rows_keeps_their_proportions() {
    let ks = || vec![kernel(Direction::Forward, 1.0, Some(200.0), Closed::Right)];
    let (out, _) = outputs(
        ks(),
        vec![op(0, Stat::Mean)],
        clock(1e9),
        &[(0.0, 1.0), (54.0, 42.0), (201.0, 99.0)],
    );
    close(out[0][0], 42.0, "one value");
    let (out, _) = outputs(
        ks(),
        vec![op(0, Stat::Mean)],
        clock(1e9),
        &[(0.0, 1.0), (54.0, 100.0), (55.0, 200.0), (201.0, 999.0)],
    );
    // `c 2^-54 (1 − 2^-1)` against `c 2^-55 (1 − 2^-145)`.
    let (a, b) = (0.5, 0.5 * (1.0 - 2f64.powi(-145)));
    close(out[0][0], (100.0 * a + 200.0 * b) / (a + b), "two values");
}

/// WIN-2 (3): a forward mean whose window holds a value only far ahead of
/// the row and is closed by a null row: the close discounted it from the
/// row, `2^-1102`, which underflows. A mean is a ratio, the same seen from
/// any time, so the close reads it where its sums are anchored: at the
/// first row in the window with a value.
#[test]
fn a_forward_mean_closed_far_from_its_row_is_its_value() {
    let (out, _) = outputs(
        vec![kernel(Direction::Forward, 1.0, Some(1202.0), Closed::Right)],
        vec![op(0, Stat::Mean)],
        clock(1e9),
        &[(1198.0, NAN), (2300.0, 42.0), (2401.0, NAN)],
    );
    close(out[0][0], 42.0, "row 0");
    // The window's oldest rows null, and two values: 42 over `[2300, 2302)`
    // and 50 over `[2302, 2400]`, masses `c (1 − 2^-2)` and `c 2^-2 (1 −
    // 2^-98)` from the first of them. The row that closes the window is
    // the one that ends the 50's interval, or one after it.
    let (a, b) = (0.75, 0.25 * (1.0 - 2f64.powi(-98)));
    let want = (42.0 * a + 50.0 * b) / (a + b);
    for last in [vec![(2401.0, NAN)], vec![(2350.0, NAN), (2401.0, 7.0)]] {
        let mut rows = vec![(1198.0, NAN), (1199.0, NAN), (2300.0, 42.0), (2302.0, 50.0)];
        rows.extend(last);
        let (out, _) = outputs(
            vec![kernel(Direction::Forward, 1.0, Some(1202.0), Closed::Right)],
            vec![op(0, Stat::Mean)],
            clock(1e9),
            &rows,
        );
        close(out[0][0], want, "two values, row 0");
    }
}

/// WIN-2 (1), (2): a quantile's weights are its rows' discounts, which past
/// 1074 half-lives underflow seen from a row that is not theirs. Its value
/// depends only on their proportions, so the same rows moved together give
/// the same quantile.
#[test]
fn a_quantile_far_from_its_anchor_is_the_quantile_of_the_rows_moved_close() {
    let backward = |rows: &[(f64, f64)]| {
        outputs(
            vec![
                kernel(Direction::Backward, 1.0, Some(2000.0), Closed::Right),
                kernel(Direction::Backward, 1.0, None, Closed::Right),
            ],
            vec![op(0, Stat::Quantile), op(1, Stat::Quantile)],
            clock(1e9),
            rows,
        )
        .0
    };
    // Row 3 holds 42 alone in its window, its newest rows null.
    let out = backward(&[(0.0, 99.0), (10.0, 42.0), (1100.0, NAN), (2001.0, NAN)]);
    assert!(!out[1][3].is_nan(), "the unbounded quantile");
    assert_eq!(out[0][3].to_bits(), out[1][3].to_bits(), "{:?}", out);
    let forward = |rows: &[(f64, f64)]| {
        outputs(
            vec![kernel(Direction::Forward, 1.0, Some(5000.0), Closed::Right)],
            vec![op(0, Stat::Quantile)],
            clock(1e9),
            rows,
        )
        .0
    };
    // Row 1 sees 100, 100 and 300 at weights 4 : 2 : 1, 1199 half-lives on;
    // the last row closes every window before it.
    let xs = [1.0, 2.0, 100.0, 100.0, 300.0, 7.0];
    let rows = |t: [f64; 6]| t.into_iter().zip(xs).collect::<Vec<_>>();
    let far = forward(&rows([0.0, 1.0, 1200.0, 1201.0, 1202.0, 1e4]));
    let near = forward(&rows([0.0, 1.0, 2.0, 3.0, 4.0, 1e4]));
    assert!(!near[0][1].is_nan());
    assert_eq!(
        far[0][1].to_bits(),
        near[0][1].to_bits(),
        "{far:?} {near:?}"
    );
}

/// WIN-8: null rows discount a mean's sums, numerator and mass alike, so
/// the mean is the same on every null row, however far: 42 over `(−∞, 0]`
/// and 50 over `(0, 1]` weigh the same from any later row, a mean of 46
/// and a variance of 16, 32 with the count correction. The sums sank into
/// the subnormals, the mean went to 33, then null.
// Confirmed; the fix moves outputs past 5 ulps or the state's layout, so
// it waits on the user (PLAN 578d, WIN-8), and this test fails until it lands.
#[test]
#[should_panic(expected = "against")]
fn a_mean_over_null_rows_holds_however_far_they_run() {
    let rows: Vec<(f64, f64)> = (0..1100)
        .map(|i| {
            (
                f64::from(i),
                [42.0, 50.0].get(i as usize).copied().unwrap_or(NAN),
            )
        })
        .collect();
    let k = || vec![kernel(Direction::Backward, 1.0, None, Closed::Right)];
    let (out, _) = outputs(
        k(),
        vec![op(0, Stat::Mean), op(0, Stat::Var), op(0, Stat::Std)],
        clock(1e9),
        &rows,
    );
    for (i, ((m, v), s)) in out[0].iter().zip(&out[1]).zip(&out[2]).enumerate().skip(1) {
        close(*m, 46.0, &format!("mean at row {i}"));
        close(*v, 32.0, &format!("variance at row {i}"));
        close(*s, 32f64.sqrt(), &format!("std at row {i}"));
    }
}

/// WIN-8 (2): under `"left"` a row reads its window without itself, so
/// at 540 and at 1080 the window holds 42 and 50 alone, 539 half-lives and
/// more back: the variance went null at 540, the mean at 1080.
// Confirmed; the fix moves outputs past 5 ulps or the state's layout, so
// it waits on the user (PLAN 578d, WIN-1 and WIN-8), and this test fails until it lands.
#[test]
#[should_panic(expected = "against")]
fn a_left_closed_window_far_from_its_rows_keeps_their_proportions() {
    let k = || vec![kernel(Direction::Backward, 1.0, Some(5000.0), Closed::Left)];
    let ops = || vec![op(0, Stat::Mean), op(0, Stat::Var)];
    let (out, _) = outputs(
        k(),
        ops(),
        clock(1e9),
        &[(0.0, 42.0), (1.0, 50.0), (540.0, 60.0), (1080.0, 70.0)],
    );
    close(out[0][2], 46.0, "mean at 540");
    close(out[1][2], 32.0, "variance at 540");
    let (out, _) = outputs(
        k(),
        ops(),
        clock(1e9),
        &[(0.0, 42.0), (1.0, 50.0), (1080.0, 70.0)],
    );
    close(out[0][2], 46.0, "mean at 1080");
}

/// WIN-10, WIN-4, WIN-5: a half-life near the largest double is a kernel
/// `check` takes, under which rows a few units apart weigh alike. A mean's
/// sum `m·x`, the unbounded mass `h / ln 2` and the variance's `V1²`
/// overflowed on ordinary values, and the outputs came out infinite or
/// null. Unbounded, the first value is held from before the stretch, a
/// mass `m₀ = h / ln 2` against the 1 of each later row: a mean of 10, and
/// a count-corrected variance `q V1 / 2e` of `500 (m₀ + 2) / (4 m₀ + 2)`,
/// 125. Read at 2 over a window of 10 the first value is held over `[−8,
/// 0]`: masses 8, 1 and 1, a mean of 13 and a variance of `410 · 10 / 34`.
// Confirmed; the fix moves outputs past 5 ulps or the state's layout, so
// it waits on the user (PLAN 578d, WIN-4, WIN-5 and WIN-10), and this test fails until it lands.
#[test]
#[should_panic(expected = "against")]
fn a_half_life_near_the_largest_double_weighs_its_rows_alike() {
    for h in [1e150, 1e200, 1e210, 1e300, 1e308, 1.5e308, f64::MAX] {
        let (out, core) = outputs(
            vec![
                kernel(Direction::Backward, h, None, Closed::Right),
                kernel(Direction::Backward, h, Some(10.0), Closed::Right),
            ],
            vec![
                op(0, Stat::Mean),
                op(1, Stat::Mean),
                op(0, Stat::Var),
                op(1, Stat::Std),
            ],
            clock(1e9),
            &[(0.0, 10.0), (1.0, 20.0), (2.0, 30.0)],
        );
        core.check().unwrap();
        close(out[0][2], 10.0, &format!("h {h}: unbounded mean"));
        close(out[1][2], 13.0, &format!("h {h}: windowed mean"));
        close(out[2][2], 125.0, &format!("h {h}: unbounded variance"));
        close(
            out[3][2],
            (410.0 * 10.0 / 34.0f64).sqrt(),
            &format!("h {h}: windowed std"),
        );
    }
}

/// WIN-9: a number clock whose every step is finite can run past the
/// largest double from its stretch's first row; the policy time was then
/// infinite, every later output null, and the state saved there refused as
/// damaged at its load. The row that takes it there is refused by name; a
/// step of the same size from a nearer origin is not.
#[test]
fn a_clock_past_the_largest_double_from_its_origin_is_refused() {
    let k = vec![kernel(Direction::Backward, 10.0, Some(50.0), Closed::Right)];
    let mut core = Windows::new(k, vec![op(0, Stat::Mean)], clock(1e308)).unwrap();
    let push = |core: &mut Windows, t: f64| {
        let group = core.group(None);
        core.push(&RowIn {
            group,
            clock: Some(ClockValue::F64(t)),
            session: None,
            values: &[1.0],
            accept: true,
            lists: &[],
        })
    };
    push(&mut core, -1e308).unwrap();
    push(&mut core, 0.0).unwrap();
    core.check().unwrap();
    let err = push(&mut core, 1e308).expect_err("refused");
    assert!(
        matches!(err, Refusal::Span { seq: 2, origin: ClockValue::F64(o), now: ClockValue::F64(n) }
            if o == -1e308 && n == 1e308),
        "{err:?}"
    );
    // From an origin at 0 the same clock runs to 1e308 and on.
    let k = vec![kernel(Direction::Backward, 10.0, Some(50.0), Closed::Right)];
    let mut core = Windows::new(k, vec![op(0, Stat::Mean)], clock(1e308)).unwrap();
    for t in [0.0, 1e308, 1.5e308] {
        push(&mut core, t).unwrap();
    }
    core.check().unwrap();
}
