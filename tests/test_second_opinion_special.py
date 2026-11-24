"""The core's special functions against ``scipy.special`` (PLAN 558, slice A).

Rust tests may not call scipy, so the native module exposes each function
elementwise (``_rheon.special_function``) for this file; the Rust side holds
the closed forms and identities (``crates/rheon-core/src/special/tests.rs``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

TIER = "essential"


def _mp_gamma_q_inv(a: float, q: float) -> float:
    """The x with Q(a, x) = q at 50 digits: a secant solve of ln Q(a, e^t) =
    ln q from scipy's value, checked to leave a relative residual of 10⁻³⁰."""
    import mpmath as mp

    guess = _gamma_guess(a, q)
    if not (np.isfinite(guess) and guess > 0):
        # The root lies below the smallest double (the far lower tail of a
        # small shape), where no value is normal: the points the gap skips.
        return float("nan")
    with mp.workdps(50):
        g = lambda t: mp.log(mp.gammainc(a, mp.exp(t), mp.inf, regularized=True)) - mp.log(q)  # noqa: E731
        t = mp.findroot(g, mp.log(mp.mpf(guess)), solver="secant", tol=mp.mpf(10) ** -40)
        x = mp.exp(t)
        residual = mp.gammainc(a, x, mp.inf, regularized=True) / q - 1
        assert abs(residual) < mp.mpf(10) ** -30, (a, q, residual)
        return float(x)


def _gamma_guess(a: float, q: float) -> float:
    from scipy import special as sc

    return float(sc.gammainccinv(a, q))


def _mp_beta_inc_c(a: float, b: float, x: float) -> float:
    """1 − I_x(a, b) at 160 digits: the upper tail, where its cancellation at
    50 digits reads zero for values near 1e-55."""
    import mpmath as mp

    with mp.workdps(160):
        return float(mp.betainc(a, b, x, 1, regularized=True))


class TestSpecialFunctionsAreScipys:
    """The core's special functions (``rheon_core::special``, PLAN 558) against
    ``scipy.special`` over the domains a distribution fit reads: Student's t's
    ``I_x(nu/2, 1/2)`` for ``nu`` in ``[2.05, 300]``, the generalized error
    distribution's ``P(1/nu, x)`` and ``P(3/nu, x)`` for ``nu`` in ``[1.01,
    500]``, and general shapes from 0.3 to 150.

    Each bound is the worst gap measured on these fixed grids (x86-64 Linux,
    scipy 1.18.1, 2026-11-23), rounded up, in units in the last place of
    scipy's value. A function of ``x`` is judged against its condition
    number ``x f(x) / F(x)``, the relative change half an ulp of ``x`` makes
    to ``F``, where that is above 1 (a CDF's far tail, where neither side can
    do better); an inverse against ``min(F, 1 - F) / (x f(x))``. ``ln Γ``,
    ``ψ`` and ``ln B`` cross zero, so they are judged absolutely, in ``ε``
    times ``max(1, |value|)``.

    ``gamma_q_inv`` and ``beta_inc_c`` are held to 50- and 160-digit mpmath
    references, rheon's own bounds; scipy is their second opinion at a bound
    that covers its gap on each platform (scipy's values differ by platform,
    so a bound set at one platform's maximum does not hold on another).

    Where a bound is wide the gap is scipy's: ``gammaincc`` near ``a = 1/2,
    x ≈ 1`` (to 380 half-ulps), ``betaln`` with one large argument (cephes
    forms it from two ``ln Γ`` of size 600; 256 ε), and ``ndtr``'s far
    lower tail (``exp(-x²)`` with ``x²`` rounded). Each was checked once
    against a 50-digit evaluation of the defining series, where the core's
    value was within 4 half-ulps; ``test_the_definitions_at_fifty_digits``
    holds the core to that standard at a few points. ``ln Γ`` and ``ψ`` are
    ``garch``'s, kept to the bit, whose recurrence sums twenty logarithms
    below an argument of 20 (an absolute error of up to 100 ε there)."""

    NU_T = np.linspace(2.05, 300.0, 400)
    NU_G = np.geomspace(1.01, 500.0, 300)
    SHAPES = np.array([0.3, 0.5, 1.0, 2.5, 7.0, 40.0, 150.0])
    P = np.concatenate([np.geomspace(1e-12, 0.1, 40), np.linspace(0.1, 0.9, 17)])

    @staticmethod
    def _call(name: str, a: Any, b: Any, x: Any) -> tuple[np.ndarray, ...]:
        from rheon import _rheon as native

        shape = np.broadcast_shapes(np.shape(a), np.shape(b), np.shape(x))
        a, b, x = (np.broadcast_to(v, shape).ravel().astype(float) for v in (a, b, x))
        got = np.array(native.special_function(name, a.tolist(), b.tolist(), x.tolist()))
        return a, b, x, got

    @staticmethod
    def _worst(got: np.ndarray, want: np.ndarray, cond: np.ndarray | None = None) -> float:
        """The largest gap in ulps of ``want`` (divided by ``cond`` where it is
        above 1), over the points scipy gives a normal value at; no point may be
        non-finite where scipy's is finite."""
        assert np.all(np.isfinite(got) | ~np.isfinite(want))
        ok = np.isfinite(want) & (np.abs(want) > 1e-300)
        gap = np.abs(got[ok] - want[ok]) / np.spacing(np.abs(want[ok]))
        if cond is not None:
            gap = gap / np.maximum(1.0, np.abs(cond[ok]))
        return float(gap.max())

    @staticmethod
    def _worst_abs(got: np.ndarray, want: np.ndarray) -> float:
        assert np.all(np.isfinite(got))
        return float(
            np.max(np.abs(got - want) / (np.finfo(float).eps * np.maximum(1.0, np.abs(want))))
        )

    def test_log_gamma_and_its_derivatives(self) -> None:
        from scipy import special as sc

        x = np.concatenate(
            [
                np.geomspace(1e-3, 1e3, 2000),
                self.NU_T / 2,
                (self.NU_T + 1) / 2,
                1 / self.NU_G,
                3 / self.NU_G,
                5 / self.NU_G,
            ]
        )
        _, _, _, got = self._call("ln_gamma", 0, 0, x)
        assert self._worst_abs(got, sc.gammaln(x)) <= 100.0  # measured 94.2
        _, _, _, got = self._call("digamma", 0, 0, x)
        assert self._worst_abs(got, sc.digamma(x)) <= 10.0  # measured 8.8
        _, _, _, got = self._call("trigamma", 0, 0, x)
        assert self._worst(got, sc.polygamma(1, x)) <= 4.0  # measured 4

    def test_incomplete_gamma_and_its_inverses(self) -> None:
        from scipy import special as sc
        from scipy import stats

        shapes = np.concatenate([1 / self.NU_G, 3 / self.NU_G, [0.5, 1, 2, 5, 20, 50, 150]])
        a, x = np.meshgrid(shapes, np.geomspace(1e-6, 700.0, 300))
        for name, oracle, bound in (
            ("gamma_p", sc.gammainc, 56.0),  # measured 52.0
            ("gamma_q", sc.gammaincc, 176.0),  # measured 169.9, scipy's gap
        ):
            aa, _, xx, got = self._call(name, a, 0, x)
            want = oracle(aa, xx)
            with np.errstate(all="ignore"):
                cond = xx * stats.gamma.pdf(xx, aa) / want
            assert self._worst(got, want, cond) <= bound, name
        a, p = np.meshgrid(
            np.concatenate([1 / self.NU_G[::5], 3 / self.NU_G[::5], [0.5, 1, 2, 5, 20, 50, 150]]),
            self.P,
        )
        for name, oracle, tail, bound in (
            ("gamma_p_inv", sc.gammaincinv, sc.gammainc, 72.0),  # measured 65.4
            # scipy's second opinion; its gap varies by platform (167 here,
            # 170 on macOS), so its bound covers both and rheon's does not.
            ("gamma_q_inv", sc.gammainccinv, sc.gammaincc, 180.0),
        ):
            aa, _, pp, got = self._call(name, a, 0, p)
            want = oracle(aa, pp)
            with np.errstate(all="ignore"):
                cond = tail(aa, want) / (want * stats.gamma.pdf(want, aa))
            assert self._worst(got, want, cond) <= bound, name
        # rheon's own bound, against a 50-digit mpmath inverse: 20.0 measured on
        # every third point of this grid, where the worst one is.
        aa, _, pp, got = self._call("gamma_q_inv", a.ravel()[::3], 0, p.ravel()[::3])
        ref = np.array([_mp_gamma_q_inv(float(x), float(q)) for x, q in zip(aa, pp, strict=True)])
        with np.errstate(all="ignore"):
            cond = sc.gammaincc(aa, ref) / (ref * stats.gamma.pdf(ref, aa))
        assert self._worst(got, ref, cond) <= 24.0, "gamma_q_inv against mpmath"

    def test_incomplete_beta_and_its_inverse(self) -> None:
        from scipy import special as sc
        from scipy import stats

        tails = np.geomspace(1e-8, 0.5, 120)
        a_t, x_t = np.meshgrid(self.NU_T / 2, np.concatenate([tails, 1 - tails]))
        a_g, b_g, x_g = np.meshgrid(self.SHAPES, self.SHAPES, np.geomspace(1e-6, 0.999, 60))
        for name, oracle, bound in (
            ("beta_inc", sc.betainc, 48.0),  # measured 8.5 (t), 41.0 (general)
            ("beta_inc_c", sc.betaincc, 240.0),  # scipy's second opinion: 221 on macOS
        ):
            for a, b, x in ((a_t, 0.5, x_t), (a_g, b_g, x_g)):
                aa, bb, xx, got = self._call(name, a, b, x)
                want = oracle(aa, bb, xx)
                with np.errstate(all="ignore"):
                    cond = xx * stats.beta.pdf(xx, aa, bb) / want
                assert self._worst(got, want, cond) <= bound, name
        # rheon's own bound, against 160-digit mpmath: 6.6 (t) and 84.8 (general)
        # measured on these grids, subsampled for time. scipy is exact there.
        for a, b, x in (
            (a_t[::8, ::3], 0.5, x_t[::8, ::3]),
            (a_g[::2, ::2], b_g[::2, ::2], x_g[::2, ::2]),
        ):
            aa, bb, xx, got = self._call("beta_inc_c", a, b, x)
            ref = np.array(
                [
                    _mp_beta_inc_c(float(p), float(q), float(r))
                    for p, q, r in zip(aa, bb, xx, strict=True)
                ]
            )
            with np.errstate(all="ignore"):
                cond = xx * stats.beta.pdf(xx, aa, bb) / ref
            assert self._worst(got, ref, cond) <= 96.0, "beta_inc_c against mpmath"
        aa, bb, _, got = self._call("ln_beta", a_g[..., 0], b_g[..., 0], 0)
        assert self._worst_abs(got, sc.betaln(aa, bb)) <= 260.0  # measured 257.5, scipy's
        a_i, p_i = np.meshgrid(self.NU_T[::8] / 2, self.P)
        a_j, b_j, p_j = np.meshgrid(self.SHAPES, self.SHAPES, self.P)
        for a, b, p in ((a_i, 0.5, p_i), (0.5, a_i, p_i), (a_j, b_j, p_j)):
            aa, bb, pp, got = self._call("beta_inc_inv", a, b, p)
            want = sc.betaincinv(aa, bb, pp)
            with np.errstate(all="ignore"):
                f = sc.betainc(aa, bb, want)
                cond = np.minimum(f, 1 - f) / (want * stats.beta.pdf(want, aa, bb))
            assert self._worst(got, want, cond) <= 128.0  # measured 128.0, 120.0, 67.0

    def test_the_normal(self) -> None:
        from scipy import special as sc
        from scipy import stats

        z = np.linspace(-37.0, 8.0, 3000)
        _, _, zz, got = self._call("norm_cdf", 0, 0, z)
        want = sc.ndtr(zz)
        cond = np.abs(zz) * stats.norm.pdf(zz) / want
        assert self._worst(got, want, cond) <= 4.0  # passes at 4
        p = np.concatenate([np.geomspace(1e-300, 0.5, 3000), 1 - np.geomspace(1e-16, 0.5, 300)])
        _, _, pp, got = self._call("ndtri", 0, 0, p)
        assert self._worst(got, sc.ndtri(pp)) <= 2.0  # measured 2

    def test_the_definitions_at_fifty_digits(self) -> None:
        """``P(a, x) = xᵃ e⁻ˣ Σₙ xⁿ / Γ(a + n + 1)``, ``I_x(a, b) = xᵃ / (a
        B(a, b)) Σₙ ((1 - b)ₙ / n!) xⁿ a / (a + n)`` and ``ln B``, summed in
        50-digit decimal arithmetic with ``ln Γ`` from Stirling's series (Bernoulli
        numbers exact) past 40: the points where scipy's own gap is the wide one,
        and a spread of others."""
        from decimal import Decimal, getcontext
        from fractions import Fraction

        getcontext().prec = 50
        bern = [Fraction(0)] * 31
        acc = [Fraction(0)] * 31
        for m in range(31):
            acc[m] = Fraction(1, m + 1)
            for j in range(m, 0, -1):
                acc[j - 1] = j * (acc[j - 1] - acc[j])
            bern[m] = acc[0]
        pi = Decimal("3.14159265358979323846264338327950288419716939937510")

        def lng(x: Decimal) -> Decimal:
            shift = Decimal(0)
            while x < 40:
                shift += x.ln()
                x += 1
            s = (x - Decimal("0.5")) * x.ln() - x + (2 * pi).ln() / 2
            for k in range(1, 15):
                b = bern[2 * k]
                s += (
                    Decimal(b.numerator)
                    / Decimal(b.denominator)
                    / (2 * k * (2 * k - 1) * x ** (2 * k - 1))
                )
            return s - shift

        def lower_gamma(a: Decimal, x: Decimal) -> Decimal:
            term = (a * x.ln() - x - lng(a + 1)).exp()
            total, n = term, 1
            while term > Decimal(10) ** -45 * total:
                term = term * x / (a + n)
                total, n = total + term, n + 1
            return total

        def lower_beta(a: Decimal, b: Decimal, x: Decimal) -> Decimal:
            front = (a * x.ln() - (lng(a) + lng(b) - lng(a + b))).exp() / a
            total, c, n = Decimal(0), Decimal(1), 0
            while True:
                t = c * x**n * a / (a + n)
                total += t
                if n > 5 and abs(t) < Decimal(10) ** -45 * abs(total):
                    return front * total
                c, n = c * (n + 1 - b) / (n + 1), n + 1

        def gap(got: float, want: Decimal) -> float:
            return float(abs(Decimal(got) - want) / Decimal(np.spacing(abs(float(want)))))

        worst = 0.0
        for a, x in [
            (0.509053, 0.719928),
            (0.509053, 1.08338),
            (0.5, 0.72),
            (0.2, 0.3),
            (0.9, 1.4),
            (0.002, 0.01),
            (2.5, 3.0),
            (30.0, 25.0),
        ]:
            p = lower_gamma(Decimal(a), Decimal(x))
            worst = max(worst, gap(self._call("gamma_p", a, 0, x)[3][0], p))
            worst = max(worst, gap(self._call("gamma_q", a, 0, x)[3][0], 1 - p))
        for a, b, x in [
            (0.3, 150.0, 0.0022887148585017387),
            (0.3, 40.0, 0.00179474),
            (0.5, 138.426, 0.00387949705927),
            (2.5, 7.0, 0.2),
            (1.025, 0.5, 0.3),
            (15.0, 0.5, 0.5),
        ]:
            i = lower_beta(Decimal(a), Decimal(b), Decimal(x))
            worst = max(worst, gap(self._call("beta_inc", a, b, x)[3][0], i))
            worst = max(worst, gap(self._call("beta_inc_c", a, b, x)[3][0], 1 - i))
        for a, b in [(150.0, 0.3), (40.0, 0.3), (0.3, 0.5), (7.0, 2.5)]:
            want = lng(Decimal(a)) + lng(Decimal(b)) - lng(Decimal(a) + Decimal(b))
            got = self._call("ln_beta", a, b, 0)[3][0]
            assert abs(Decimal(got) - want) <= Decimal(4 * np.finfo(float).eps) * max(1, abs(want))
        assert worst <= 20.0  # measured 17.6, at I_x(0.3, 150)'s complement
