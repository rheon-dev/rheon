//! The functions the root-finder's tests invert, written as a user would:
//! Black-76's call price in its volatility, and a cubic. Each returns its
//! value and a bound on its rounding error, carried beside the value by a
//! running error analysis (Higham, *Accuracy and Stability of Numerical
//! Algorithms*, 2nd ed., §3.3): every operation adds `u·|result|` to the
//! bounds propagated from its operands, to first order, and the final bound
//! is doubled to cover the second-order terms.
//!
//! Shared by the example `root_black76` (which `tests/test_second_opinion_root.py`
//! holds to scipy's `brentq`) and the Rust tests of `src/root.rs`.

/// The unit roundoff, `2⁻⁵³`.
const U: f64 = f64::EPSILON / 2.0;

/// The smallest subnormal, `2⁻¹⁰⁷⁴`: an underflowing result can lose up
/// to this much absolutely (half of it rounded to nearest; `erfc` far in
/// its tail returns 0 for a value below it), so every operation adds it.
const ETA: f64 = f64::from_bits(1);

/// A math library call's relative error, in units of `u`: the bound the
/// library's own oracles use (`src/oracle.rs`, `LIBM`).
const LIBM: f64 = 6.0;

/// `Φ(z) = erfc(−z/√2) / 2`, the standard normal CDF, accurate in both
/// tails (`1 − erf` loses the left one), from the `libm` crate: the same
/// bits on every platform. The one place the tests name it.
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z / std::f64::consts::SQRT_2)
}

/// A value and a bound on its absolute error.
#[derive(Clone, Copy)]
struct R {
    v: f64,
    e: f64,
}

impl R {
    fn exact(v: f64) -> Self {
        Self { v, e: 0.0 }
    }

    fn add(self, o: R) -> R {
        let v = self.v + o.v;
        R {
            v,
            e: self.e + o.e + U * v.abs() + ETA,
        }
    }

    fn sub(self, o: R) -> R {
        let v = self.v - o.v;
        R {
            v,
            e: self.e + o.e + U * v.abs() + ETA,
        }
    }

    fn mul(self, o: R) -> R {
        let v = self.v * o.v;
        R {
            v,
            e: self.v.abs() * o.e + o.v.abs() * self.e + self.e * o.e + U * v.abs() + ETA,
        }
    }

    fn div(self, o: R) -> R {
        let v = self.v / o.v;
        let room = o.v.abs() - o.e;
        let e = if room > 0.0 {
            (self.e + v.abs() * o.e) / room + U * v.abs() + ETA
        } else {
            f64::INFINITY
        };
        R { v, e }
    }

    fn ln(self) -> R {
        let v = self.v.ln();
        let room = self.v - self.e;
        let e = if room > 0.0 {
            self.e / room + LIBM * U * v.abs()
        } else {
            f64::INFINITY
        };
        R { v, e }
    }

    fn sqrt(self) -> R {
        let v = self.v.sqrt();
        let e = if self.e == 0.0 {
            U * v
        } else if v > 0.0 {
            self.e / v + U * v
        } else {
            self.e.sqrt()
        };
        R { v, e }
    }

    /// `Φ`: the argument's own error and the rounding of `−z/√2` (two
    /// roundings of `z`), each moving `Φ` by at most the density's largest
    /// value on the interval they span, and the call's own error.
    fn norm_cdf(self) -> R {
        let v = norm_cdf(self.v);
        let dz = self.e + 2.0 * U * self.v.abs();
        let near = (self.v.abs() - dz).max(0.0);
        let density = libm::exp(-0.5 * near * near) / (2.0 * std::f64::consts::PI).sqrt();
        R {
            v,
            e: dz * density * 1.01 + LIBM * U * v + ETA,
        }
    }

    fn done(self) -> (f64, f64) {
        (self.v, 2.0 * self.e)
    }
}

/// Black-76's call price at volatility `sigma`, less `price`:
/// `df·(F·Φ(d₁) − K·Φ(d₁ − v)) − price`, `v = σ√τ`,
/// `d₁ = (ln(F/K) + v²/2) / v`. Increasing in `sigma` for `τ > 0`.
pub fn call_less(sigma: f64, fwd: f64, strike: f64, tau: f64, df: f64, price: f64) -> (f64, f64) {
    let v = R::exact(sigma).mul(R::exact(tau).sqrt());
    let lq = R::exact(fwd).div(R::exact(strike)).ln();
    let d1 = lq.add(R::exact(0.5).mul(v).mul(v)).div(v);
    let d2 = d1.sub(v);
    let call = R::exact(df).mul(
        R::exact(fwd)
            .mul(d1.norm_cdf())
            .sub(R::exact(strike).mul(d2.norm_cdf())),
    );
    call.sub(R::exact(price)).done()
}

/// `x³ + c·x − t`, increasing for `c ≥ 0`.
pub fn cubic_less(x: f64, c: f64, t: f64) -> (f64, f64) {
    let x = R::exact(x);
    x.mul(x)
        .mul(x)
        .add(R::exact(c).mul(x))
        .sub(R::exact(t))
        .done()
}
