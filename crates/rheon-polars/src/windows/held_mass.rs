//! A backward row's held mass: the time its value is held over, the partner
//! of its discount ([`mass_held_to`]), and the power-of-two mass unit for
//! half-lives from `2^511` up (review 2026-11-23, WIN-4, WIN-5, WIN-10).
use super::*;

/// The distance, in half-lives, from which a held mass is taken from the
/// discount's `exp2` instead of `exp_m1`: a hundredth of a half-life.
pub(super) const HELD_MASS_FROM: f64 = 1e-2;

/// Biased exponent of `2^511`, the smallest half-life from which masses are
/// kept in a power-of-two unit `2^k` so `h / 2^k < 2^511`. Below `2^511`,
/// `(h / ln 2)^2 < f64::MAX` and `(h / ln 2) * 1e100 < f64::MAX`, so the unit
/// is `1.0` and no bit moves.
const MASS_SCALE_EXP: u64 = 511 + 1023;

/// `2^511`: see [`MASS_SCALE_EXP`].
pub(super) const MASS_SCALE_FROM: f64 = f64::from_bits(MASS_SCALE_EXP << 52);

/// Threshold on `t * ln 2 / h` below which `∫_0^t 2^(-s/h) ds = t * (1 - arg/2 + ...)`
/// rounds to `t` within `0.25` ulp (`arg / 2 < 2^-54`), avoiding subnormal
/// precision loss and underflow in `exp_m1` at extreme half-lives or tiny steps.
pub(super) const MASS_LINEAR_BELOW: f64 = 0.5 * f64::EPSILON;

impl KernelDef {
    /// Power-of-two factor `2^-k` for a finite half-life at least `2^511`,
    /// placing `half_life * scale` in `[2^510, 2^511)` (`scale` in `[2^-513, 2^-1]`).
    #[inline]
    fn mass_scale_huge(&self) -> f64 {
        let exp = (self.half_life.to_bits() >> 52) & 0x7ff;
        debug_assert!((MASS_SCALE_EXP..2047).contains(&exp));
        // `k = exp - (MASS_SCALE_EXP - 1) = exp - 1533` in `1..=513`.
        // `scale = 2^-k` has biased exponent `1023 - k = 2556 - exp` in `510..=1022`.
        f64::from_bits((2556 - exp) << 52)
    }

    /// Power-of-two factor `2^-k` converting clock units into this kernel's
    /// mass unit: `1.0` for every half-life below `2^511` (and for `inf`),
    /// and otherwise `2^-(floor(log2(h)) - 510)` so `h * mass_scale` lies in
    /// `[2^510, 2^511)`. Derived purely from `half_life`, so no state byte is
    /// added.
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline]
    pub(super) fn mass_scale(&self) -> f64 {
        let h = self.half_life;
        if h >= MASS_SCALE_FROM && h.is_finite() {
            self.mass_scale_huge()
        } else {
            1.0
        }
    }

    /// [`Self::mass`] for `half_life >= 2^511` (both `inf` and huge finite).
    #[cold]
    #[inline(never)]
    pub(super) fn mass_extreme(&self, t: f64) -> f64 {
        if self.half_life.is_infinite() {
            t
        } else {
            self.mass_huge(t)
        }
    }

    /// [`Self::mass_extreme`] through one slot of a [`DecayMemo`].
    #[cold]
    #[inline(never)]
    pub(super) fn mass_extreme_in(&self, t: f64, slot: &mut (u64, f64)) -> f64 {
        if self.half_life.is_infinite() {
            return t;
        }
        let bits = t.to_bits();
        if slot.0 == bits {
            return slot.1;
        }
        let v = self.mass_huge(t);
        *slot = (bits, v);
        v
    }

    /// [`Self::mass_extreme`] through `memo.cut`.
    #[cold]
    #[inline(never)]
    pub(super) fn mass_extreme_cut(&self, t: f64, memo: &mut DecayMemo) -> f64 {
        if self.half_life.is_infinite() {
            return t;
        }
        let bits = t.to_bits();
        let slot = &mut memo.cut.get_or_insert_with(Default::default)[(bits & 3) as usize];
        if slot.0 == bits {
            return slot.1;
        }
        let v = self.mass_huge(t);
        *slot = (bits, v);
        v
    }

    /// [`Self::mass_normal`] for a finite half-life at least `2^511`: the
    /// decayed mass in this kernel's power-of-two unit, or `t * scale` where
    /// `t * ln 2 / h < 2^-53`.
    #[cold]
    #[inline(never)]
    pub(super) fn mass_huge(&self, t: f64) -> f64 {
        let h = self.half_life;
        let scale = self.mass_scale_huge();
        let arg = t * std::f64::consts::LN_2 / h;
        if arg < MASS_LINEAR_BELOW {
            return t * scale;
        }
        -(h * scale / std::f64::consts::LN_2) * (-arg).exp_m1()
    }

    /// `∫_0^t λ^s ds` in clock units (not divided by the kernel's mass unit),
    /// for `Stat::Rate`, whose numerator is an unscaled decayed sum. For any
    /// finite `t >= 0`, the integral is at most `t <= f64::MAX`, so it never
    /// overflows even when `h / ln 2` does.
    #[inline]
    pub(super) fn clock_mass(&self, t: f64) -> f64 {
        if t <= 0.0 {
            0.0
        } else if self.half_life.is_infinite() {
            t
        } else {
            self.clock_mass_finite(t)
        }
    }

    /// [`Self::clock_mass`] through one slot of a [`DecayMemo`].
    #[inline]
    pub(super) fn clock_mass_slot(&self, t: f64, slot: &mut (u64, f64)) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        if self.half_life.is_infinite() {
            return t;
        }
        if slot.0 == t.to_bits() {
            return slot.1;
        }
        let v = self.clock_mass_finite(t);
        *slot = (t.to_bits(), v);
        v
    }

    #[inline]
    fn clock_mass_finite(&self, t: f64) -> f64 {
        let h = self.half_life;
        let arg = t * std::f64::consts::LN_2 / h;
        if arg < MASS_LINEAR_BELOW {
            return t;
        }
        let c = h / std::f64::consts::LN_2;
        if c.is_finite() {
            -c * (-arg).exp_m1()
        } else {
            self.clock_mass_overflow(arg)
        }
    }

    /// [`Self::clock_mass_finite`] when `h / ln 2` overflows (`h > f64::MAX * ln 2`)
    /// and `arg >= 2^-53`: halve `h` before dividing by `ln 2` (`(f64::MAX * 0.5) / ln 2
    /// ≈ 1.30e308 < f64::MAX`), then double the product back into clock units.
    #[cold]
    #[inline(never)]
    fn clock_mass_overflow(&self, arg: f64) -> f64 {
        (-(self.half_life * 0.5 / std::f64::consts::LN_2) * (-arg).exp_m1()) * 2.0
    }

    /// The mass a row holds its value over, `d` units of time: its discount's
    /// partner, through `memo`'s slot for it (the cut's mass of the same
    /// distance is [`Self::mass_in`]'s, and keeps `exp_m1`).
    #[inline]
    pub(super) fn mass_held_in(&self, d: f64, memo: &mut DecayMemo) -> f64 {
        if d <= 0.0 {
            return 0.0;
        }
        if self.half_life >= MASS_SCALE_FROM {
            return self.mass_held_extreme(d, memo);
        }
        let (bits, v) = memo.held;
        if bits == d.to_bits() {
            return v;
        }
        let v = self.mass_held_normal(d, memo);
        memo.held = (d.to_bits(), v);
        v
    }

    /// [`Self::mass_normal`] of the distance a row holds its value over (for a
    /// finite half-life below `2^511`). From `HELD_MASS_FROM` half-lives up it
    /// is `c (1 - d)` from the very `d = 2^(-x)` the row's discount is, taken
    /// through the discount's own slot, so a row's step and its mass cost one
    /// `exp2` between them where they ask for the same distance; the two forms
    /// agree to a few ulps there (`1 - d` loses at most `2^-52 / (1 - d)` of
    /// the mass, 1.6e-14 at the cut). Below it `1 - d` is too coarse (doubles
    /// next to 1 are 1.1e-16 apart) and `exp_m1` is kept. The choice is on `x`
    /// (an IEEE division and a compare), never on a library result, so every
    /// platform takes the same form for the same row.
    #[inline]
    fn mass_held_normal(&self, d: f64, memo: &mut DecayMemo) -> f64 {
        let h = self.half_life;
        if d / h >= HELD_MASS_FROM {
            return (h / std::f64::consts::LN_2) * (1.0 - self.discount_in(d, memo));
        }
        self.mass_normal(d)
    }

    #[cold]
    #[inline(never)]
    fn mass_held_extreme(&self, d: f64, memo: &mut DecayMemo) -> f64 {
        if self.half_life.is_infinite() {
            return d;
        }
        let (bits, v) = memo.held;
        if bits == d.to_bits() {
            return v;
        }
        let h = self.half_life;
        let v = if d / h >= HELD_MASS_FROM {
            let scale = self.mass_scale_huge();
            (h * scale / std::f64::consts::LN_2) * (1.0 - self.discount_in(d, memo))
        } else {
            self.mass_huge(d)
        };
        memo.held = (d.to_bits(), v);
        v
    }
}
