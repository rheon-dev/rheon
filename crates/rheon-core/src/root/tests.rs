//! The root-finder's guards, its tolerance and its bound, each on both sides
//! of its threshold; the bound over hundreds of thousands of brackets; and
//! Black-76's implied volatility against bisection to the last double
//! (`bisect_root`, below). scipy's `brentq` holds the same solves in
//! `tests/test_second_opinion_root.py`, through the example `root_black76`.

use super::*;
use crate::SplitMix64;

#[allow(dead_code)]
#[path = "../../examples/root_black76/black76.rs"]
mod black76;

/// The computed function's sign change on `[lo, hi]` to the last double, by
/// bisection alone: `f(lo) < 0 < f(hi)` is required, and the bracket is
/// halved at `a + (b − a)/2` until its ends are adjacent doubles, or an
/// evaluation is exactly 0 (returned as `(x, x)`). It shares no step with
/// the solver: no regula falsi, no projection, no tolerance, and no
/// error bound, so it is an oracle for a function computed without
/// rounding, or for the computed function's own sign change.
fn bisect_root(mut f: impl FnMut(f64) -> f64, lo: f64, hi: f64) -> (f64, f64) {
    let (mut a, mut b) = (lo, hi);
    assert!(
        f(a) < 0.0 && f(b) > 0.0,
        "bisection needs f({a}) < 0 < f({b})"
    );
    while a.next_up() < b {
        let m = a + 0.5 * (b - a);
        if m <= a || m >= b {
            break;
        }
        let y = f(m);
        assert!(!y.is_nan(), "f({m}) is NaN");
        if y == 0.0 {
            return (m, m);
        }
        if y < 0.0 {
            a = m;
        } else {
            b = m;
        }
    }
    (a, b)
}

fn exact(f: impl Fn(f64) -> f64) -> impl FnMut(f64) -> Bounded {
    move |x| Bounded::exact(f(x))
}

fn tol(xtol: f64, rtol: f64) -> RootTolerance {
    RootTolerance::new(xtol, rtol).unwrap()
}

// --- the tolerance ---------------------------------------------------------

#[test]
fn the_tolerance_is_refused_at_each_edge_and_taken_just_inside() {
    assert!(RootTolerance::new(0.0, MIN_RTOL).is_err());
    assert!(RootTolerance::new(-1e-12, MIN_RTOL).is_err());
    assert!(RootTolerance::new(f64::NAN, MIN_RTOL).is_err());
    assert!(RootTolerance::new(f64::INFINITY, MIN_RTOL).is_err());
    assert!(RootTolerance::new(f64::from_bits(1), MIN_RTOL).is_ok());
    assert!(RootTolerance::new(1e300, MIN_RTOL).is_ok());
    assert!(RootTolerance::new(1e-12, MIN_RTOL.next_down()).is_err());
    assert!(RootTolerance::new(1e-12, MIN_RTOL).is_ok());
    assert!(RootTolerance::new(1e-12, f64::NAN).is_err());
    assert!(RootTolerance::new(1e-12, f64::INFINITY).is_err());
    let d = RootTolerance::default();
    assert_eq!((d.xtol(), d.rtol()), (2e-12, 4.0 * 2f64.powi(-52)));
}

#[test]
fn eps_is_xtol_where_the_bracket_holds_zero_and_grows_with_its_nearer_end() {
    let t = tol(1e-12, 1e-10);
    assert_eq!(t.eps(-1.0, 2.0), 1e-12);
    assert_eq!(t.eps(0.0, 2.0), 1e-12);
    assert_eq!(t.eps(-3.0, 0.0), 1e-12);
    assert_eq!(t.eps(2.0, 5.0), 1e-12 + 1e-10 * 2.0);
    assert_eq!(t.eps(-5.0, -3.0), 1e-12 + 1e-10 * 3.0);
}

#[test]
fn the_step_count_is_bisections_plus_one_on_each_side_of_a_power_of_two() {
    let eps = 0.375;
    for k in 0..60 {
        let width = 2.0 * eps * 2f64.powi(k);
        assert_eq!(itp_steps(width, eps), k as u32 + N0, "k = {k}");
        assert_eq!(itp_steps(width.next_up(), eps), k as u32 + 1 + N0);
    }
    assert_eq!(itp_steps(1e-300, eps), N0);
    // The widest bracket at the smallest tolerance: finite, about 2,100.
    let most = itp_steps(f64::MAX, f64::from_bits(1));
    assert!((2000..2200).contains(&most), "{most}");
}

// --- the stop and the bound ------------------------------------------------

#[test]
fn a_bracket_at_the_stop_takes_no_step_and_one_wider_takes_one() {
    // `[0, hi]` holds 0, so `ε = xtol`; the stop is `2ε` plus eight machine
    // epsilons of `hi`.
    let e = 2f64.powi(-20);
    let t = tol(e, MIN_RTOL);
    let f = exact(move |x| x - e / 4.0);
    let at = find_root(f, 0.0, 2.0 * e * (1.0 + 8.0 * f64::EPSILON), true, &t);
    assert_eq!(at.evaluations, 2);
    assert!((at.root.unwrap() - e / 4.0).abs() <= e * (1.0 + 1e-12));
    let f = exact(move |x| x - e / 4.0);
    let over = find_root(f, 0.0, 2.0 * e * (1.0 + 16.0 * f64::EPSILON), true, &t);
    assert_eq!(over.evaluations, 3);
    assert!((over.root.unwrap() - e / 4.0).abs() <= e);
}

#[test]
fn reaching_the_bound_without_the_tolerance_is_reported_never_hidden() {
    let f = exact(|x| x - 0.3);
    let short = solve(f, 0.0, 1.0, true, 1e-12, 0);
    assert_eq!(short.root, Err(Withheld::BoundReached));
    assert_eq!((short.evaluations, short.bound), (2, 4));
    let n = itp_steps(1.0, 1e-12);
    let enough = solve(exact(|x| x - 0.3), 0.0, 1.0, true, 1e-12, n);
    assert!(enough.root.is_ok());
    assert!(enough.evaluations <= enough.bound);
    assert_eq!(enough.bound, n + 4);
}

#[test]
fn an_exact_zero_is_a_root_at_an_end_or_inside_and_needs_a_zero_error_bound() {
    let t = RootTolerance::default();
    let lo = find_root(exact(|x| x), 0.0, 1.0, true, &t);
    assert_eq!(
        (lo.root, lo.lo, lo.hi, lo.evaluations),
        (Ok(0.0), 0.0, 0.0, 2)
    );
    let hi = find_root(exact(|x| x - 1.0), 0.0, 1.0, true, &t);
    assert_eq!((hi.root, hi.lo, hi.hi), (Ok(1.0), 1.0, 1.0));
    // Regula falsi lands on 0.5 at once.
    let mid = find_root(exact(|x| x - 0.5), 0.0, 1.0, true, &t);
    assert_eq!((mid.root, mid.evaluations), (Ok(0.5), 3));
    // A zero that rounding could have made: at the end the exact root may
    // lie outside the bracket, so it is withheld; inside, the certificate's
    // two points fix it.
    let noisy = |x: f64| Bounded::new(x, 1e-20);
    let r = find_root(noisy, 0.0, 1.0, true, &t);
    assert_eq!(r.root, Err(Withheld::NotDetermined));
    let r = find_root(|x: f64| Bounded::new(x - 0.5, 1e-20), 0.0, 1.0, true, &t);
    assert_eq!(r.root, Ok(0.5));
    assert!(r.lo < 0.5 && r.hi > 0.5 && r.hi - r.lo <= 4.0 * t.eps(0.0, 1.0) + 1e-15);
    assert_eq!(r.evaluations, 5);
}

// --- the guards ------------------------------------------------------------

#[test]
fn a_bracket_that_is_not_an_interval_is_refused_before_any_evaluation() {
    let t = RootTolerance::default();
    let cases = [
        (f64::NAN, 1.0, Withheld::InputMissing),
        (0.0, f64::INFINITY, Withheld::InputMissing),
        (f64::NEG_INFINITY, 0.0, Withheld::InputMissing),
        (1.0, 1.0, Withheld::BracketInvalid),
        (2.0, 1.0, Withheld::BracketInvalid),
        (-1e308, 1e308, Withheld::BracketInvalid),
    ];
    for (lo, hi, why) in cases {
        let r = find_root(exact(|x| x), lo, hi, true, &t);
        assert_eq!(
            (r.root, r.evaluations, r.bound),
            (Err(why), 0, 0),
            "[{lo}, {hi}]"
        );
    }
    // One double wide is an interval: no step, and the certificate decides.
    let r = find_root(exact(|x| x - 1.0), 1.0, 1f64.next_up(), true, &t);
    assert_eq!(r.root, Ok(1.0));
    let r = find_root(
        exact(|x| x - 1.0 - f64::EPSILON / 2.0),
        1.0,
        1f64.next_up(),
        true,
        &t,
    );
    assert_eq!((r.root, r.evaluations), (Ok(1.0), 2));
    // Just inside the overflow: a finite width.
    assert!(
        find_root(exact(|x| x), -8e307, 8e307, true, &t)
            .root
            .is_ok()
    );
}

#[test]
fn no_sign_change_names_the_side_the_target_lies_on_in_either_direction() {
    let t = RootTolerance::default();
    type Case = (fn(f64) -> f64, bool, Withheld);
    let cases: [Case; 4] = [
        (|x| x - 5.0, true, Withheld::AboveRange),
        (|x| x + 5.0, true, Withheld::BelowRange),
        (|x| 5.0 - x, false, Withheld::BelowRange),
        (|x| -5.0 - x, false, Withheld::AboveRange),
    ];
    for (f, inc, why) in cases {
        let r = find_root(exact(f), 0.0, 1.0, inc, &t);
        assert_eq!((r.root, r.evaluations), (Err(why), 2));
    }
    // Touching the target at an end is a root, not out of range.
    assert_eq!(
        find_root(exact(|x| x - 1.0), 0.0, 1.0, true, &t).root,
        Ok(1.0)
    );
    assert_eq!(
        find_root(exact(|x| 1.0 - x), 0.0, 1.0, false, &t).root,
        Ok(1.0)
    );
}

#[test]
fn a_sign_change_against_the_declared_direction_is_refused() {
    let t = RootTolerance::default();
    let r = find_root(exact(|x| 0.5 - x), 0.0, 1.0, true, &t);
    assert_eq!(r.root, Err(Withheld::WrongDirection));
    let r = find_root(exact(|x| x - 0.5), 0.0, 1.0, false, &t);
    assert_eq!(r.root, Err(Withheld::WrongDirection));
    // Zero at one end and the wrong sign at the other.
    let r = find_root(exact(|x| -x), 0.0, 1.0, true, &t);
    assert_eq!(r.root, Err(Withheld::WrongDirection));
    let r = find_root(exact(|x| 1.0 - x), 0.0, 1.0, true, &t);
    assert_eq!(r.root, Err(Withheld::WrongDirection));
    // The same functions, declared the way they go, are solved.
    assert!(
        find_root(exact(|x| 0.5 - x), 0.0, 1.0, false, &t)
            .root
            .is_ok()
    );
    assert!(find_root(exact(|x| -x), 0.0, 1.0, false, &t).root.is_ok());
    // Zero at both ends: a constant, no unique root.
    let r = find_root(exact(|_| 0.0), 0.0, 1.0, true, &t);
    assert_eq!(r.root, Err(Withheld::NotDetermined));
}

#[test]
fn nan_is_a_domain_error_at_an_end_or_a_step_and_an_infinity_is_a_sign() {
    let t = RootTolerance::default();
    let r = find_root(exact(f64::ln), -1.0, 2.0, true, &t);
    assert_eq!((r.root, r.evaluations), (Err(Withheld::NotFinite), 2));
    let hole = |x: f64| {
        if x > 0.3 && x < 0.7 {
            f64::NAN
        } else {
            x - 0.6
        }
    };
    let r = find_root(exact(hole), 0.0, 1.0, true, &t);
    assert_eq!((r.root, r.evaluations), (Err(Withheld::NotFinite), 3));
    assert_eq!((r.lo, r.hi), (0.0, 1.0));
    // ln 0 = −∞ at the lower end, 1/0 = +∞ at the upper one.
    let r = find_root(exact(f64::ln), 0.0, 4.0, true, &t);
    assert!((r.root.unwrap() - 1.0).abs() <= t.eps(0.0, 4.0));
    let r = find_root(exact(|x| 1.0 / (1.0 - x) - 2.0), 0.0, 1.0, true, &t);
    assert!((r.root.unwrap() - 0.5).abs() <= t.eps(0.0, 1.0));
    // A certificate point in the hole.
    let edge = |x: f64| Bounded::new(if x > 0.5 { f64::NAN } else { x - 0.5 }, 1.0);
    let r = find_root(edge, 0.0, 0.5 + 1e-12, true, &t);
    assert_eq!(r.root, Err(Withheld::NotFinite));
}

#[test]
fn a_step_outside_the_ends_values_beyond_both_bounds_is_not_monotone() {
    let t = RootTolerance::default();
    // Ends at −0.7 and 0.3, each value within 0.01; the first step lands
    // at 0.5, inside the bump.
    let bump = |height: f64| {
        move |x: f64| Bounded::new(if x > 0.4 && x < 0.6 { height } else { x - 0.7 }, 0.01)
    };
    let r = find_root(bump(0.33), 0.0, 1.0, true, &t);
    assert_eq!((r.root, r.evaluations), (Err(Withheld::NotMonotone), 3));
    // Within the bounds of the upper end's value: not provably outside.
    let r = find_root(bump(0.305), 0.0, 1.0, true, &t);
    assert_ne!(r.root, Err(Withheld::NotMonotone));
    let r = find_root(bump(-0.75), 0.0, 1.0, true, &t);
    assert_eq!(r.root, Err(Withheld::NotMonotone));
    let r = find_root(bump(-0.705), 0.0, 1.0, true, &t);
    assert_ne!(r.root, Err(Withheld::NotMonotone));
    // A sign change inside the ends' values passes unseen: the check costs
    // no evaluation, so it cannot find every non-monotone function.
    let r = find_root(bump(0.2), 0.0, 1.0, true, &t);
    assert!((r.root.unwrap() - 0.4).abs() < 1e-11);
}

#[test]
fn the_certificate_passes_where_the_signs_outgrow_the_rounding_and_withholds_where_not() {
    let t = tol(1e-9, MIN_RTOL);
    let eps = t.eps(-1.0, 1.0);
    let c = 0.123_456_789;
    let with = |err: f64| move |x: f64| Bounded::new(x - c, err);
    // No rounding: the final bracket's own ends certify it.
    let r = find_root(with(0.0), -1.0, 1.0, true, &t);
    assert!(r.hi - r.lo <= 2.0 * eps);
    // An error the final ends may not clear but `m ± 2ε` must (they are at
    // least ε from the root): the wider bracket certifies it.
    let r = find_root(with(0.5 * eps), -1.0, 1.0, true, &t);
    let x = r.root.unwrap();
    assert!((x - c).abs() <= eps, "{x}");
    assert!(r.lo < c && c < r.hi && r.hi - r.lo <= 4.0 * eps * (1.0 + 1e-9));
    // An error past what `m ± 2ε` can clear (they are at most 3ε away).
    let r = find_root(with(3.1 * eps), -1.0, 1.0, true, &t);
    assert_eq!(r.root, Err(Withheld::NotDetermined));
    assert!(r.evaluations <= r.bound);
    // An infinite error bound, or a NaN one, makes no sign certain.
    let r = find_root(with(f64::NAN), -1.0, 1.0, true, &t);
    assert_eq!(r.root, Err(Withheld::NotDetermined));
    // A negative bound is read as its magnitude.
    let r = find_root(with(-3.1 * eps), -1.0, 1.0, true, &t);
    assert_eq!(r.root, Err(Withheld::NotDetermined));
}

#[test]
fn every_reason_has_its_own_name() {
    let names: std::collections::HashSet<_> = Withheld::ALL.iter().map(|w| w.name()).collect();
    assert_eq!(names.len(), Withheld::ALL.len());
    assert_eq!(Withheld::NotDetermined.name(), "not_determined");
    assert_eq!(Withheld::BelowRange.name(), "below_range");
}

// --- the direction -----------------------------------------------------------

#[test]
fn a_decreasing_function_is_its_negation_to_the_bit() {
    let t = RootTolerance::default();
    let mut rng = SplitMix64::new(563);
    for _ in 0..2000 {
        let (c, t3) = (rng.uniform() * 3.0, (rng.uniform() - 0.3) * 20.0);
        let up = find_root(
            |x: f64| {
                let (v, e) = black76::cubic_less(x, c, t3);
                Bounded::new(v, e)
            },
            -3.0,
            4.0,
            true,
            &t,
        );
        let down = find_root(
            |x: f64| {
                let (v, e) = black76::cubic_less(x, c, t3);
                Bounded::new(-v, e)
            },
            -3.0,
            4.0,
            false,
            &t,
        );
        assert_eq!(format!("{up:?}"), format!("{down:?}"));
        assert!(up.root.is_ok(), "{up:?}");
    }
}

// --- the bound, over many brackets --------------------------------------------

/// One random monotone family, computed with no rounding the solve must
/// allow for (each is monotone as computed, so its own values are the
/// exact function), its root `c` and a bracket around it, at a random
/// tolerance: the root, the bracket and the tolerance each span the doubles'
/// range.
fn random_case(rng: &mut SplitMix64) -> (usize, f64, f64, f64, f64, RootTolerance) {
    let span = |rng: &mut SplitMix64, lo: f64, hi: f64| 10f64.powf(lo + (hi - lo) * rng.uniform());
    let signed = |rng: &mut SplitMix64, v: f64| if rng.uniform() < 0.5 { -v } else { v };
    let c = match rng.next_u64() % 4 {
        0 => 0.0,
        1 => rng.uniform() * 2.0 - 1.0,
        _ => {
            let v = span(rng, -300.0, 300.0);
            signed(rng, v)
        }
    };
    let width = |rng: &mut SplitMix64| {
        if rng.uniform() < 0.5 && c != 0.0 {
            c.abs() * span(rng, -15.0, 3.0)
        } else {
            span(rng, -300.0, 300.0)
        }
    };
    let (wl, wr) = (width(rng), width(rng));
    let (lo, hi) = (c - wl, c + wr);
    let k = span(rng, -3.0, 3.0) / wl.max(wr);
    let t = tol(span(rng, -320.0, 0.0), MIN_RTOL * span(rng, 0.0, 12.0));
    ((rng.next_u64() % 6) as usize, c, k, lo, hi, t)
}

fn family(which: usize, c: f64, k: f64) -> impl Fn(f64) -> f64 {
    move |x: f64| {
        let d = x - c;
        match which {
            0 => d,
            1 => {
                if d < 0.0 {
                    -1.0
                } else if d > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            2 => d * d.abs(),
            3 => d * d * d,
            // `k` may overflow on a narrow bracket: `∞·0` is not the sign.
            4 if d == 0.0 => 0.0,
            4 => libm::tanh(k * d),
            _ => libm::atan(d) + libm::cbrt(d),
        }
    }
}

/// `n` random brackets: the bound always holds and is never reached, every
/// root is within `ε` (or a double's spacing) of the computed function's
/// sign change by its definition and by bisection, and on the step family,
/// where regula falsi has nothing to go on, the solve comes within `n₀` of
/// the bound, so the bound is no looser than the paper's.
fn the_bound_holds_over(n: usize, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let (mut solved, mut tightest) = (0usize, u32::MAX);
    for i in 0..n {
        let (which, c, k, lo, hi, t) = random_case(&mut rng);
        let f = family(which, c, k);
        let r = find_root(exact(&f), lo, hi, true, &t);
        if !(lo < hi && (hi - lo).is_finite()) {
            assert!(r.root.is_err() && r.evaluations == 0);
            continue;
        }
        let ctx = format!("case {i}: family {which}, c {c:e}, [{lo:e}, {hi:e}], {t:?}: {r:?}");
        assert!(r.evaluations <= r.bound, "{ctx}");
        if f(lo) == 0.0 && f(hi) == 0.0 {
            // An underflowing square or cube: zero at both ends as computed.
            assert_eq!(r.root, Err(Withheld::NotDetermined), "{ctx}");
            continue;
        }
        let x = r.root.expect(&ctx);
        solved += 1;
        let eps = t.eps(lo, hi);
        // The stated `ε + 5·eps_mach·|root|`, `ε` itself rounded, and two
        // subnormal spacings where the doubles are that sparse.
        let slack =
            eps * (1.0 + 4.0 * f64::EPSILON) + 5.0 * f64::EPSILON * x.abs() + f64::from_bits(2);
        let (below, above) = ((x - slack).max(lo), (x + slack).min(hi));
        assert!(f(below) <= 0.0 && f(above) >= 0.0, "{ctx}");
        if f(lo) < 0.0 && f(hi) > 0.0 {
            // Unless bisection lands on an exact zero, which may be another
            // point of a run of zeros (an underflowing cube).
            let (a, b) = bisect_root(&f, lo, hi);
            assert!(
                a == b || (x >= a - slack && x <= b + slack),
                "{ctx}: bisection [{a:e}, {b:e}]"
            );
        }
        if which == 1 && r.hi - r.lo > 0.0 {
            tightest = tightest.min(r.bound - r.evaluations);
        }
    }
    assert!(solved > n * 9 / 10, "{solved} of {n}");
    // Two ends, `n½` halvings, no certificate: `n_max + 4 − (n½ + 2) = n₀ + 2`.
    assert_eq!(tightest, N0 + 2);
}

#[test]
fn the_stated_bound_holds_on_twenty_thousand_random_brackets() {
    the_bound_holds_over(20_000, 1);
}

#[test]
#[ignore = "extended: half a million random brackets in a debug build"]
fn the_stated_bound_holds_on_half_a_million_random_brackets() {
    the_bound_holds_over(500_000, 2);
}

// --- Black-76 ------------------------------------------------------------------

/// A seeded Black-76 grid: forward 100, expiries from a day to two years,
/// log-moneyness within three standard deviations (`±3σ√τ`), volatilities from 5% to 150%, the
/// price at that volatility. Each solve's bound holds; every certified root
/// lies, to a double, in its certified bracket and beside the computed
/// function's sign change by bisection; and the rows withheld are few (441
/// of 20,000, measured), each one where the rounding of the price moves the
/// root by about the tolerance or more.
#[test]
fn black76_implied_volatility_agrees_with_bisection_where_certified() {
    let t = RootTolerance::default();
    let (lo, hi) = (1e-4, 5.0);
    let eps = t.eps(lo, hi);
    let mut rng = SplitMix64::new(76);
    let n = 20_000;
    let (mut withheld, mut worst) = (0usize, 0.0f64);
    for i in 0..n {
        let tau = 10f64.powf(-2.562 + 2.863 * rng.uniform());
        let sigma = 0.05 + 1.45 * rng.uniform();
        let z = 3.0 * (2.0 * rng.uniform() - 1.0);
        let strike = 100.0 * libm::exp(z * sigma * tau.sqrt());
        let df = libm::exp(-0.03 * tau);
        let price = black76::call_less(sigma, 100.0, strike, tau, df, 0.0).0;
        let g = |s: f64| black76::call_less(s, 100.0, strike, tau, df, price);
        let r = find_root(
            |s| {
                let (v, e) = g(s);
                Bounded::new(v, e)
            },
            lo,
            hi,
            true,
            &t,
        );
        let ctx = format!("row {i}: K {strike}, tau {tau}, sigma {sigma}: {r:?}");
        assert!(r.evaluations <= r.bound, "{ctx}");
        assert!(r.evaluations <= 44 + 2, "{ctx}");
        let vega = df * 100.0 * tau.sqrt() * (-0.5 * d1(strike, tau, sigma).powi(2)).exp()
            / (2.0 * std::f64::consts::PI).sqrt();
        match r.root {
            Ok(x) => {
                assert!(
                    r.lo <= x && x <= r.hi && r.hi - r.lo <= 4.0 * eps + 1e-15,
                    "{ctx}"
                );
                let (a, b) = bisect_root(|s| g(s).0, lo, hi);
                assert!(
                    a <= r.hi.next_up() && b >= r.lo.next_down(),
                    "{ctx}: [{a}, {b}]"
                );
                if vega > 1.0 {
                    worst = worst.max((x - sigma).abs());
                }
            }
            Err(why) => {
                withheld += 1;
                assert_eq!(why, Withheld::NotDetermined, "{ctx}");
                // Withheld only where the rounding moves the root by about
                // `ε` or more: `m ± 2ε` lie at least `ε` from it, so a
                // slope there of `vega` clears an error below `ε·vega`.
                // Measured at least 1.16 times `ε` over this grid.
                let spread = g(sigma).1 / vega;
                assert!(
                    spread > 0.5 * eps,
                    "{ctx}: the rounding moves it {spread:e}"
                );
            }
        }
    }
    assert!(withheld * 20 < n, "{withheld} of {n} withheld");
    assert!(worst < 1e-10, "{worst}");
}

fn d1(strike: f64, tau: f64, sigma: f64) -> f64 {
    let v = sigma * tau.sqrt();
    ((100.0 / strike).ln() + 0.5 * v * v) / v
}
