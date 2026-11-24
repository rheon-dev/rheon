//! A guarded root-finder for a monotone function of one variable: the ITP
//! method (Oliveira and Takahashi, "An enhancement of the bisection method
//! average performance preserving minmax optimality", ACM TOMS 47(1), 2021),
//! stopped by a tolerance, with a worst case stated before the first step,
//! and a root reported only where the function's own rounding fixes it.
//!
//! [`find_root`] solves `f(x) = 0` on a bracket `[lo, hi]` for an `f` the
//! caller declares increasing or decreasing there. A caller inverting
//! `f(x) = target` passes `x ↦ f(x) − target`. Each evaluation returns the
//! value and a bound on its rounding error ([`Bounded`]); the solve reads a
//! value's sign as certain only where the value is larger than its bound.
//!
//! # The step
//!
//! With `h = f` (or `−f` where `f` is declared decreasing), so that
//! `h(a) < 0 < h(b)` on the bracket `[a, b]` holding the sign change, and
//! `ε = xtol + rtol · min(|lo|, |hi|)` (`xtol` alone where `[lo, hi]` holds
//! 0), the `j`-th step (from 0) evaluates
//!
//! ```text
//! x½   = (a + b) / 2                         the midpoint
//! x_f  = a − h(a)·(b − a) / (h(b) − h(a))    regula falsi
//! δ    = κ₁ (b − a)²,  κ₁ = 0.2 / (hi − lo)  the truncation
//! x_t  = x_f + sign(x½ − x_f)·δ   if δ ≤ |x½ − x_f|,  else x½
//! r_j  = ε·2^(n_max − j) − (b − a)/2         the projection's radius
//! x    = x_t   if |x_t − x½| ≤ r_j,  else x½ − sign(x½ − x_f)·r_j
//! ```
//!
//! and keeps the half of `[a, b]` whose ends' signs differ. Since `x` is
//! within `r_j` of the midpoint, the bracket after step `j` is at most
//! `ε·2^(n_max − j)` wide, so after `n_max = n½ + n₀` steps it is at most
//! `2ε`, where `n½ = ⌈log₂((hi − lo) / 2ε)⌉` is bisection's count and
//! `n₀ = 1`. Where the regula falsi point is not finite (an infinite value
//! at an end) the truncation falls back to the midpoint.
//!
//! In floating point the midpoint and each step's point are rounded to a
//! double, so a bracket can come out up to about one unit in the last place
//! of its ends wider than that bound; each such excess is halved by every
//! later step, so together they stay within a few units. The radius is
//! rounded down (by four units in the last place of `ε·2^(n_max − j)` and
//! two of the bracket's larger end), and the stop allows eight machine
//! epsilons of the bracket's larger end beyond `2ε`. Without that allowance
//! a solve of a cubic on `[−3, 4]` ran its `n_max` steps and stopped a few
//! units in the last place of its ends wider than `2ε`.
//!
//! # When it stops
//!
//! At the first of: the bracket `[a, b]` is at most
//! `2ε + 8·eps_mach·max(|a|, |b|)` wide; its ends are adjacent doubles
//! (where the doubles are sparser than the allowance, as among the
//! subnormals); or an evaluation is exactly 0. The answer is the bracket's
//! midpoint, within `ε + 5·eps_mach·|root|` (half the allowance, and the
//! midpoint's own rounding), or one double's spacing, of a sign change of
//! the computed function.
//!
//! **The bound is a consequence of the tolerance, not a cap.** The solve
//! takes at most [`bound`](RootReport::bound) `= n_max + 4` evaluations: the
//! two at the bracket's ends, `n_max` steps, and at most two for the
//! certificate below. Since `xtol > 0`, `n½` is finite for every finite
//! bracket: at most about 2,100 (a bracket as wide as the doubles at a
//! subnormal `xtol`). Reaching `n_max` steps without meeting the tolerance
//! would be a defect of this solver, and it is reported as
//! [`Withheld::BoundReached`] for a caller to raise as an error; the tests
//! drive hundreds of thousands of brackets, the root, the bracket's width
//! and the tolerance each spanning the doubles' range, and assert it never
//! happens.
//!
//! # The certificate
//!
//! Where the computed function is flat, its sign near the root is rounding
//! noise: every point within `η / |f′|` of the root, `η` the rounding, is a
//! root of the computed function. So a root is reported only where the final
//! bracket's ends have signs larger than their error bounds: then the exact
//! function changes sign inside it. Where an end's is not, the points
//! `m − 2ε` and `m + 2ε` around the midpoint `m` (at least one double beyond
//! the bracket's ends, and never past `[lo, hi]`) are evaluated, and if
//! their signs are certain the root is reported with that wider bracket;
//! otherwise it is withheld as [`Withheld::NotDetermined`]. An evaluation
//! that is exactly 0 with a zero error bound is a root as it stands.
//!
//! The same rule holds before the first step: the ends' values are read as
//! out of range, or against the declared direction, only where their signs
//! (or, for the direction, their order) are certain; a target within the
//! rounding of an end's value is [`Withheld::NotDetermined`], since it may
//! lie inside the bracket.
//!
//! # What is withheld
//!
//! A root is withheld, never clipped to the bracket or guessed: see
//! [`Withheld`] for each reason. Nothing here allocates, and every division
//! is guarded: the regula falsi ratio by the bracket's ends having opposite
//! signs, `κ₁` by `hi > lo`.

/// A value of the function and a bound on its absolute rounding error: the
/// exact function's value at that point lies in `value ± error`.
///
/// An error of NaN reads as infinite (no sign is certain); a negative one
/// as its magnitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounded {
    /// The computed value.
    pub value: f64,
    /// A bound on `|computed − exact|`, at least 0.
    pub error: f64,
}

impl Bounded {
    /// A value with its error bound.
    pub fn new(value: f64, error: f64) -> Self {
        Self { value, error }
    }

    /// A value computed with no rounding, such as an exact polynomial.
    pub fn exact(value: f64) -> Self {
        Self { value, error: 0.0 }
    }

    /// The error bound as the solve reads it: NaN as infinite, a negative
    /// bound as its magnitude.
    fn bound(self) -> f64 {
        if self.error.is_nan() {
            f64::INFINITY
        } else {
            self.error.abs()
        }
    }
}

/// The smallest relative tolerance accepted: four machine epsilons, below
/// which a bracket around a large root cannot narrow as asked.
pub const MIN_RTOL: f64 = 4.0 * f64::EPSILON;

/// The default absolute tolerance: `2e-12`, in `x`'s units.
pub const DEFAULT_XTOL: f64 = 2e-12;

/// The solve's slack over bisection, in steps (the paper's `n₀`).
const N0: u32 = 1;

/// A solve's tolerance, checked when it is built: `xtol` finite and above
/// 0, `rtol` finite and at least [`MIN_RTOL`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootTolerance {
    xtol: f64,
    rtol: f64,
}

impl Default for RootTolerance {
    /// `xtol = 2e-12` and `rtol` four machine epsilons.
    fn default() -> Self {
        Self {
            xtol: DEFAULT_XTOL,
            rtol: MIN_RTOL,
        }
    }
}

impl RootTolerance {
    /// The tolerance, or the reason it is refused.
    pub fn new(xtol: f64, rtol: f64) -> Result<Self, String> {
        if !(xtol.is_finite() && xtol > 0.0) {
            return Err(format!(
                "xtol must be finite and above 0, got {xtol}: at 0 a bracket around a root \
                 at 0 could never narrow to it"
            ));
        }
        if !(rtol.is_finite() && rtol >= MIN_RTOL) {
            return Err(format!(
                "rtol must be finite and at least {MIN_RTOL:e} (four machine epsilons), got \
                 {rtol}: below that a bracket around a large root cannot narrow as asked"
            ));
        }
        Ok(Self { xtol, rtol })
    }

    /// The absolute tolerance.
    pub fn xtol(&self) -> f64 {
        self.xtol
    }

    /// The relative tolerance.
    pub fn rtol(&self) -> f64 {
        self.rtol
    }

    /// `ε` for the bracket `[lo, hi]`: `xtol + rtol · min(|lo|, |hi|)`, or
    /// `xtol` alone where the bracket holds 0.
    pub fn eps(&self, lo: f64, hi: f64) -> f64 {
        if lo > 0.0 || hi < 0.0 {
            self.xtol + self.rtol * lo.abs().min(hi.abs())
        } else {
            self.xtol
        }
    }
}

/// Why a root is withheld.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Withheld {
    /// A bracket end is NaN or infinite: no interval to search.
    InputMissing,
    /// `lo >= hi`, or `hi − lo` overflows: no interval to search.
    BracketInvalid,
    /// `f` is above 0 at both ends, each beyond its error bound: the target
    /// lies below the function's values on the bracket. With a monotone
    /// function, no root anywhere in it.
    BelowRange,
    /// `f` is below 0 at both ends, each beyond its error bound: the target
    /// lies above the function's values on the bracket.
    AboveRange,
    /// The ends' signs change against the declared direction, and their
    /// values differ beyond both error bounds: the function is not what the
    /// caller declared, and a root found would be of another function.
    WrongDirection,
    /// NaN at an end or at a step: a domain error in the function. An
    /// infinity is not this case: it has a sign.
    NotFinite,
    /// A step's value lies, beyond both error bounds, outside the values at
    /// the bracket's two ends: the declared monotonicity is broken inside
    /// the bracket, so the root may not be unique. Checked at no extra
    /// evaluation, so a non-monotone function can also pass unseen.
    NotMonotone,
    /// The function's own rounding decides the answer: the certificate
    /// failed, so the root's position is not fixed to the tolerance; or the
    /// ends' values are one sign, or against the declared direction, only
    /// within their error bounds (the target may lie inside the bracket);
    /// or `f` is exactly 0 at both ends.
    NotDetermined,
    /// `n_max` steps without the tolerance met: a defect of this solver,
    /// never expected, for a caller to raise as an error.
    BoundReached,
}

impl Withheld {
    /// Every reason, in declaration order.
    pub const ALL: [Withheld; 9] = [
        Withheld::InputMissing,
        Withheld::BracketInvalid,
        Withheld::BelowRange,
        Withheld::AboveRange,
        Withheld::WrongDirection,
        Withheld::NotFinite,
        Withheld::NotMonotone,
        Withheld::NotDetermined,
        Withheld::BoundReached,
    ];

    /// The reason's name, as an output column carries it.
    pub fn name(self) -> &'static str {
        match self {
            Withheld::InputMissing => "input_missing",
            Withheld::BracketInvalid => "bracket_invalid",
            Withheld::BelowRange => "below_range",
            Withheld::AboveRange => "above_range",
            Withheld::WrongDirection => "wrong_direction",
            Withheld::NotFinite => "not_finite",
            Withheld::NotMonotone => "not_monotone",
            Withheld::NotDetermined => "not_determined",
            Withheld::BoundReached => "bound_reached",
        }
    }
}

/// A solve's outcome: the root or why it is withheld, the bracket, the
/// evaluations it took and the bound they were held to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootReport {
    /// The root, or why it is withheld.
    pub root: Result<f64, Withheld>,
    /// The certified bracket's lower end where a root is given; otherwise
    /// the last bracket's (`lo` itself where the solve stopped before a
    /// step, and NaN where the bracket was not read).
    pub lo: f64,
    /// The bracket's upper end, as `lo`.
    pub hi: f64,
    /// The evaluations of `f` the solve took.
    pub evaluations: u32,
    /// The most it could take: `n_max + 4` (0 where the bracket was refused).
    pub bound: u32,
}

/// `n_max = n½ + n₀`: the most steps the solve takes on a bracket `width`
/// wide at the tolerance `eps`, `n½` the least `k ≥ 0` with
/// `2ε·2^k ≥ width`, so that bisection's `k` halvings would leave at most
/// `2ε`. Each `2ε·2^k` is exact (a power of two scales without rounding)
/// until it overflows, which ends the count.
pub fn itp_steps(width: f64, eps: f64) -> u32 {
    let two_eps = 2.0 * eps;
    let mut k = 0u32;
    while libm::scalbn(two_eps, k as i32) < width {
        k += 1;
    }
    k + N0
}

/// The root of `f` on `[lo, hi]`, for an `f` declared increasing there
/// (`increasing = true`) or decreasing, to the tolerance `tol`; see the
/// module's documentation for the step, the stop, the bound and the
/// certificate. A decreasing function is solved as its negation, which is
/// exact, so `f` declared decreasing gives the same bits as `−f` declared
/// increasing.
pub fn find_root(
    f: impl FnMut(f64) -> Bounded,
    lo: f64,
    hi: f64,
    increasing: bool,
    tol: &RootTolerance,
) -> RootReport {
    if !(lo.is_finite() && hi.is_finite()) {
        return refused(lo, hi, Withheld::InputMissing);
    }
    if !(lo < hi && (hi - lo).is_finite()) {
        return refused(lo, hi, Withheld::BracketInvalid);
    }
    let eps = tol.eps(lo, hi);
    solve(f, lo, hi, increasing, eps, itp_steps(hi - lo, eps))
}

/// A bracket refused before any evaluation.
fn refused(lo: f64, hi: f64, why: Withheld) -> RootReport {
    RootReport {
        root: Err(why),
        lo,
        hi,
        evaluations: 0,
        bound: 0,
    }
}

/// A value whose sign is certainly negative: below 0 by more than its
/// bound, or `−∞`.
fn surely_negative(y: f64, e: f64) -> bool {
    y < 0.0 && (y == f64::NEG_INFINITY || -y > e)
}

/// A value whose sign is certainly positive.
fn surely_positive(y: f64, e: f64) -> bool {
    y > 0.0 && (y == f64::INFINITY || y > e)
}

/// A value whose sign is certain: nonzero, and larger than its bound (or
/// infinite).
fn certain(at: End) -> bool {
    surely_negative(at.y, at.e) || surely_positive(at.y, at.e)
}

/// One end of a bracket: its point, the value of `h` there and its bound.
#[derive(Clone, Copy)]
struct End {
    x: f64,
    y: f64,
    e: f64,
}

/// The solve after the bracket is checked, `n_max` given so that a test can
/// set it below the solve's need and reach [`Withheld::BoundReached`].
fn solve(
    mut f: impl FnMut(f64) -> Bounded,
    lo: f64,
    hi: f64,
    increasing: bool,
    eps: f64,
    n_max: u32,
) -> RootReport {
    let sign = if increasing { 1.0 } else { -1.0 };
    let mut evaluations = 0u32;
    let mut h = |x: f64| {
        evaluations += 1;
        let v = f(x);
        End {
            x,
            y: sign * v.value,
            e: v.bound(),
        }
    };
    let width0 = hi - lo;
    let bound = n_max + 4;
    let low = h(lo);
    let high = h(hi);
    let outcome = classify(low, high, sign).map(|exact| match exact {
        Some(end) => certify_zero(&mut h, end, low, high, eps),
        None => iterate(&mut h, low, high, eps, n_max, width0),
    });
    let (root, a, b) = match outcome {
        Err(why) => (Err(why), lo, hi),
        Ok(found) => found,
    };
    RootReport {
        root,
        lo: a,
        hi: b,
        evaluations,
        bound,
    }
}

/// The ends' verdict before any step: a reason to withhold, an end at which
/// `h` is exactly 0 (`Some`), or a sign change to search (`None`). `sign`
/// undoes the direction's negation, so that "below" and "above" read the
/// function's own values.
fn classify(low: End, high: End, sign: f64) -> Result<Option<End>, Withheld> {
    if low.y.is_nan() || high.y.is_nan() {
        return Err(Withheld::NotFinite);
    }
    // A verdict on the ends' signs is given only where their rounding
    // cannot have made it: a target within the rounding of an end's value
    // may lie inside the bracket.
    let (fl, fh) = (sign * low.y, sign * high.y);
    let one_sign = |side: Withheld| {
        if certain(low) && certain(high) {
            Err(side)
        } else {
            Err(Withheld::NotDetermined)
        }
    };
    if fl > 0.0 && fh > 0.0 {
        return one_sign(Withheld::BelowRange);
    }
    if fl < 0.0 && fh < 0.0 {
        return one_sign(Withheld::AboveRange);
    }
    if low.y == 0.0 && high.y == 0.0 {
        return Err(Withheld::NotDetermined);
    }
    if low.y > 0.0 || high.y < 0.0 {
        return Err(if low.y - low.e > high.y + high.e {
            Withheld::WrongDirection
        } else {
            Withheld::NotDetermined
        });
    }
    Ok(if low.y == 0.0 {
        Some(low)
    } else if high.y == 0.0 {
        Some(high)
    } else {
        None
    })
}

type Found = (Result<f64, Withheld>, f64, f64);

/// The ITP steps from `h(a) < 0 < h(b)`, then the certificate.
fn iterate(
    h: &mut impl FnMut(f64) -> End,
    low: End,
    high: End,
    eps: f64,
    n_max: u32,
    width0: f64,
) -> Found {
    let (mut a, mut b) = (low, high);
    let mut j = 0u32;
    loop {
        if stopped(a.x, b.x, eps) {
            break;
        }
        if j == n_max {
            return (Err(Withheld::BoundReached), a.x, b.x);
        }
        let x = itp_point(a, b, eps, n_max - j, width0);
        j += 1;
        let at = h(x);
        if at.y.is_nan() {
            return (Err(Withheld::NotFinite), a.x, b.x);
        }
        if at.y + at.e < a.y - a.e || at.y - at.e > b.y + b.e {
            return (Err(Withheld::NotMonotone), a.x, b.x);
        }
        if at.y == 0.0 {
            return certify_zero(h, at, low, high, eps);
        }
        if at.y < 0.0 {
            a = at;
        } else {
            b = at;
        }
    }
    if surely_negative(a.y, a.e) && surely_positive(b.y, b.e) {
        return (Ok(a.x + 0.5 * (b.x - a.x)), a.x, b.x);
    }
    let m = a.x + 0.5 * (b.x - a.x);
    widened(h, m, a.x, b.x, low, high, eps)
}

/// Whether the bracket `[a, b]` meets the tolerance: at most `2ε` wide,
/// plus `8` machine epsilons of its larger end for the rounding of the
/// points the steps chose (each step's rounding is halved by every later
/// one, so they sum to a few units in the last place of the end), or two
/// adjacent doubles.
fn stopped(a: f64, b: f64, eps: f64) -> bool {
    let rounding = 8.0 * f64::EPSILON * a.abs().max(b.abs());
    b - a <= 2.0 * eps + rounding || a.next_up() >= b
}

/// The point the `j`-th step evaluates, `left = n_max − j` steps left; see
/// the module's documentation. Always strictly inside `(a, b)`.
fn itp_point(a: End, b: End, eps: f64, left: u32, width0: f64) -> f64 {
    let w = b.x - a.x;
    let half = a.x + 0.5 * w;
    // ε·2^(n_max − j) is exact; the radius is rounded down so that the
    // rounding of `half` and of the point cannot carry the next bracket past
    // it (the module's documentation).
    let scale = libm::scalbn(eps, left as i32);
    let slack = 4.0 * f64::EPSILON * scale + 2.0 * f64::EPSILON * a.x.abs().max(b.x.abs());
    let r = ((scale - 0.5 * w) - slack).max(0.0);
    // Regula falsi as a fraction of the bracket: `h(a) < 0 < h(b)`, so the
    // denominator is above 0 and the fraction in [0, 1] where both are
    // finite; an infinite end gives NaN or an end, and the midpoint.
    let t = -a.y / (b.y - a.y);
    let xf = a.x + t * w;
    let toward = if half > xf { 1.0 } else { -1.0 };
    let xt = if !(t.is_finite() && xf > a.x && xf < b.x) {
        half
    } else {
        let delta = 0.2 * w * (w / width0);
        if delta <= (half - xf).abs() {
            xf + toward * delta
        } else {
            half
        }
    };
    let x = if (xt - half).abs() <= r {
        xt
    } else {
        half - toward * r
    };
    if x > a.x && x < b.x { x } else { half }
}

/// An evaluation exactly 0 at `at`: a root as it stands with a zero error
/// bound, the certificate's wider bracket around it otherwise.
fn certify_zero(h: &mut impl FnMut(f64) -> End, at: End, low: End, high: End, eps: f64) -> Found {
    if at.e == 0.0 {
        return (Ok(at.x), at.x, at.x);
    }
    widened(h, at.x, at.x, at.x, low, high, eps)
}

/// The certificate's second chance: `h` at `m − 2ε` and `m + 2ε`, each at
/// least one double beyond `[a, b]` and clamped to `[lo, hi]` (where the
/// end's value is read again, not evaluated). The root `m` is reported with
/// that bracket where both signs are certain.
fn widened(
    h: &mut impl FnMut(f64) -> End,
    m: f64,
    a: f64,
    b: f64,
    low: End,
    high: End,
    eps: f64,
) -> Found {
    let pa = (m - 2.0 * eps).min(a.next_down());
    let pb = (m + 2.0 * eps).max(b.next_up());
    let left = if pa <= low.x { low } else { h(pa) };
    let right = if pb >= high.x { high } else { h(pb) };
    if left.y.is_nan() || right.y.is_nan() {
        return (Err(Withheld::NotFinite), left.x, right.x);
    }
    if surely_negative(left.y, left.e) && surely_positive(right.y, right.e) {
        (Ok(m), left.x, right.x)
    } else {
        (Err(Withheld::NotDetermined), left.x, right.x)
    }
}

#[cfg(test)]
mod tests;
