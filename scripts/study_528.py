"""Task 528: exact accumulators for sums of f64, shard merges, and their costs.

One entry point, three studies (``all`` runs them in turn):

* ``accumulators`` (part 1): the three accumulators below against
  ``math.fsum`` and ``fractions.Fraction``, shard invariance, state bytes.
* ``merge`` (part 2): merging the raw weighted sums a Gram would hold,
  without decay and with decay (shard clocks, rounded powers).
* ``costs`` (part 3): word, limb and floating-point operations per add,
  merge, canonicalisation and rounding, and state bytes, counted on the
  data (not timed).

Every claim is checked against a stated oracle, and the proposed tolerances
are asserted at the end: the script exits 1 if one fails.

Run: ``.venv/bin/python scripts/study_528.py [accumulators|merge|costs|all]
[--quick]`` (``all`` about 20 s; ``--quick`` about 5 s).

Three exact accumulators, each with ``add(x)``, ``merge(other)`` and
``round_to_f64()``:

* ``Shewchuk``: a non-overlapping floating-point expansion (Shewchuk 1997),
  grown by two-sum, as CPython's ``math.fsum`` keeps its partials.
* ``Kulisch``: a fixed-point two's-complement integer of 34 limbs of 64 bits
  (2176 bits), scaled by 2^1074 so every finite f64 is an integer in it.
* ``SuperAcc``: a two-level sparse superaccumulator after Neal (2015): a
  sparse "large" level keyed by the f64 exponent field, holding signed
  mantissa sums, flushed into a sparse "small" level of 64-bit chunks that
  each carry 32 bits of payload, with carry propagation at a set interval.

All three round the exact sum to the nearest f64, ties to even, so their
results equal ``float(Fraction(exact sum))``. The study checks that against
``math.fsum`` and ``fractions.Fraction``, then shards 1000-value sets at
random and merges in random order and grouping.

Python only; every input is generated from a seed; nothing is written to disk.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
import time
from fractions import Fraction

import numpy as np

SEED = 528

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
INT64_MAX = (1 << 63) - 1


def f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def same_bits(a: float, b: float) -> bool:
    return f64_bits(a) == f64_bits(b)


# The proposed tolerances, asserted: (claim, oracle, measured, tolerance, ok).
CHECKS: list[tuple[str, str, str, str, bool]] = []


def check(claim: str, oracle: str, measured: str, tolerance: str, ok: bool) -> None:
    CHECKS.append((claim, oracle, measured, tolerance, ok))


def decompose(x: float) -> tuple[int, int, int]:
    """A finite nonzero f64 as (sign, m, p): |x| = m * 2^(p - 1074), m < 2^53, p >= 0.

    p is the bit position of m's lowest bit in the integer |x| * 2^1074.
    A subnormal has p = 0 and no hidden bit, so it needs no special case.
    """
    bits = f64_bits(x)
    sign = -1 if bits >> 63 else 1
    ef = (bits >> 52) & 0x7FF
    frac = bits & ((1 << 52) - 1)
    if ef == 0:
        return sign, frac, 0
    return sign, frac | (1 << 52), ef - 1


def round_scaled_int(n: int) -> float:
    """Round n * 2^-1074 to the nearest f64, ties to even. Overflow gives +-inf.

    An exact zero gives +0.0, as math.fsum does (fsum([-0.0]) is 0.0).
    """
    if n == 0:
        return 0.0
    neg = n < 0
    a = -n if neg else n
    length = a.bit_length()
    if length <= 53:
        r = math.ldexp(a, -1074)  # exact: a fits the significand at this exponent
    else:
        shift = length - 53
        q = a >> shift
        rem = a & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
            if q == 1 << 53:
                q >>= 1
                shift += 1
        try:
            r = math.ldexp(q, shift - 1074)
        except OverflowError:
            r = math.inf
    return -r if neg else r


class _Specials:
    """Non-finite inputs propagate as three flags, which merge by OR.

    The result is NaN if a NaN was added or both infinities were; otherwise
    the infinity's sign; otherwise the finite exact sum. (math.fsum instead
    raises ValueError on inf + -inf; IEEE and this choice give NaN.)
    """

    def __init__(self) -> None:
        self.nan = False
        self.pinf = False
        self.ninf = False

    def take(self, x: float) -> bool:
        if math.isfinite(x):
            return False
        if math.isnan(x):
            self.nan = True
        elif x > 0:
            self.pinf = True
        else:
            self.ninf = True
        return True

    def merge(self, other: _Specials) -> None:
        self.nan |= other.nan
        self.pinf |= other.pinf
        self.ninf |= other.ninf

    def result(self) -> float | None:
        if self.nan or (self.pinf and self.ninf):
            return math.nan
        if self.pinf:
            return math.inf
        if self.ninf:
            return -math.inf
        return None


# --------------------------------------------------------------------------
# (a) Shewchuk expansion
# --------------------------------------------------------------------------


def fsum_round(p: list[float]) -> tuple[float, int]:
    """CPython fsum's final step, and the number of two-sums it took.

    Add from the top until a nonzero error, then correct a half-way case to
    round-half-even using the sign of the next lower component (one more
    two-sum when it applies).
    """
    steps = 0
    n = len(p)
    hi = 0.0
    if n > 0:
        n -= 1
        hi = p[n]
        lo = 0.0
        while n > 0:
            x = hi
            n -= 1
            y = p[n]
            hi = x + y
            yr = hi - x
            lo = y - yr
            steps += 1
            if lo != 0.0:
                break
        if n > 0 and ((lo < 0.0 and p[n - 1] < 0.0) or (lo > 0.0 and p[n - 1] > 0.0)):
            y = lo * 2.0
            x = hi + y
            yr = x - hi
            steps += 1
            if y == yr:
                hi = x
    return hi + 0.0, steps  # -0.0 -> +0.0 as fsum


class Shewchuk:
    """Non-overlapping expansion, smallest magnitude first (CPython fsum's partials).

    add(x) is Grow-Expansion with zero elimination: two-sum x through every
    component, keep each nonzero rounding error, the last sum on top. The
    components' bit ranges are disjoint, but neighbours need not be 53 bits
    apart, so the raw length is bounded only by the bit range (about 2,100
    one-bit components) and by the count of adds; on full-range random data
    it reaches 60 to 70. ``canonical()`` (each component the rounding of what
    is left, Shewchuk's compressed form taken to its limit) puts each component
    within half an ulp of the next, so at most 40 (2098 / 53 + 1).
    A two-sum that overflows raises OverflowError, as fsum does: the expansion
    cannot hold an intermediate above the f64 range even when the exact sum
    is finite.
    """

    def __init__(self) -> None:
        self.p: list[float] = []
        self.sp = _Specials()
        self.two_sums = 0  # two-sums performed, for the operation counts

    def add(self, x: float) -> None:
        if self.sp.take(x):
            return
        p = self.p
        self.two_sums += len(p)
        i = 0
        for y in p:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            if math.isinf(hi):
                raise OverflowError("intermediate overflow in the expansion")
            lo = y - (hi - x)
            if lo:
                p[i] = lo
                i += 1
            x = hi
        p[i:] = [x]

    def merge(self, other: Shewchuk) -> None:
        for c in other.p:
            self.add(c)
        self.sp.merge(other.sp)

    def _round(self, p: list[float]) -> float:
        return fsum_round(p)[0]

    def round_to_f64(self) -> float:
        s = self.sp.result()
        return s if s is not None else self._round(self.p)

    def canonical(self) -> tuple[float, ...]:
        """A form that depends on the exact value only: the greedy expansion.

        c1 = round(S), c2 = round(S - c1), ...: each residue is formed exactly
        by adding -c to the expansion, so the list is determined by S.
        """
        e = Shewchuk()
        e.p = list(self.p)
        out = []
        while e.p:
            r = e._round(e.p)
            if r == 0.0:
                break
            out.append(r)
            e.add(-r)
            e.p = [c for c in e.p if c != 0.0]
        return tuple(out)

    def state_bytes(self) -> int:
        return 8 * len(self.p) + 1 + 1  # components, a length byte, the flag byte


# --------------------------------------------------------------------------
# (b) Kulisch long accumulator
# --------------------------------------------------------------------------

K_LIMBS = 34  # 34 * 64 = 2176 bits >= 2098 (range) + 64 (count headroom) + 1 (sign)
K_BITS = 64 * K_LIMBS


class Kulisch:
    """Fixed-point two's complement over 34 limbs of 64 bits, value N * 2^-1074.

    add(x): |x| * 2^1074 = m << p with m < 2^53. Limb i = p // 64 takes the low
    64 bits of m << (p % 64), limb i+1 the rest (m << s < 2^117, so two limbs).
    A positive x adds with carry, a negative one subtracts with borrow; the
    carry or borrow ripples up until it is absorbed, wrapping mod 2^2176.
    Bits 2098 .. 2174 are headroom: 2^64 adds of the largest f64 cannot reach
    the sign bit. merge adds the other's 34 limbs with carry. The limbs are
    the canonical form: one value, one bit pattern.
    """

    def __init__(self) -> None:
        self.limbs = [0] * K_LIMBS
        self.sp = _Specials()
        self.ripple = 0  # carry steps beyond the two limbs written, for the report

    def add(self, x: float) -> None:
        if self.sp.take(x) or x == 0.0:
            return
        sign, m, p = decompose(x)
        i, s = divmod(p, 64)
        v = m << s
        parts = (v & MASK64, v >> 64)
        L = self.limbs
        carry = 0
        if sign > 0:
            for j in (0, 1):
                t = L[i + j] + parts[j] + carry
                L[i + j] = t & MASK64
                carry = t >> 64
            k = i + 2
            while carry and k < K_LIMBS:
                t = L[k] + 1
                L[k] = t & MASK64
                carry = t >> 64
                k += 1
                self.ripple += 1
        else:
            for j in (0, 1):
                t = L[i + j] - parts[j] - carry
                L[i + j] = t & MASK64
                carry = 1 if t < 0 else 0
            k = i + 2
            while carry and k < K_LIMBS:
                t = L[k] - 1
                L[k] = t & MASK64
                carry = 1 if t < 0 else 0
                k += 1
                self.ripple += 1

    def merge(self, other: Kulisch) -> None:
        carry = 0
        for k in range(K_LIMBS):
            t = self.limbs[k] + other.limbs[k] + carry
            self.limbs[k] = t & MASK64
            carry = t >> 64
        self.sp.merge(other.sp)

    def to_int(self) -> int:
        n = 0
        for k in reversed(range(K_LIMBS)):
            n = (n << 64) | self.limbs[k]
        if n >> (K_BITS - 1):
            n -= 1 << K_BITS
        return n

    def round_to_f64(self) -> float:
        s = self.sp.result()
        return s if s is not None else round_scaled_int(self.to_int())

    def canonical(self) -> tuple[int, ...]:
        return tuple(self.limbs)

    @staticmethod
    def state_bytes() -> int:
        return 8 * K_LIMBS + 1  # limbs and the flag byte


# --------------------------------------------------------------------------
# (c) Two-level sparse superaccumulator (after Neal 2015)
# --------------------------------------------------------------------------


class SuperAcc:
    """Two sparse levels.

    Large level: exponent field (0..2047) -> [signed sum of mantissas, count].
    Each mantissa is below 2^53, so 1024 of them stay below 2^63: an entry
    flushes into the small level when its count reaches ``large_limit``
    (at most 1024).

    Small level: chunk index c -> signed int64 holding bits 32c .. 32c+31 of
    N (N * 2^-1074 the value). A flushed entry (|sum| < 2^63, position p)
    adds signed 32-bit pieces into at most three chunks (63 + 31 bits).
    Each piece is below 2^32, so a chunk starting in [-1, 2^32) takes 2^31 - 1
    pieces before it could leave int64; carries propagate every
    ``carry_every`` flushes (at most 2^31 - 1), leaving every chunk in
    [0, 2^32) but for a top chunk of -1 on a negative sum (the minimal two's
    complement digits). The canonical form (for a saved state) flushes every
    entry and propagates: a function of N alone.
    """

    def __init__(self, large_limit: int = 1024, carry_every: int = (1 << 31) - 1) -> None:
        assert 1 <= large_limit <= 1024 and 1 <= carry_every <= (1 << 31) - 1
        self.large: dict[int, list[int]] = {}
        self.small: dict[int, int] = {}
        self.large_limit = large_limit
        self.carry_every = carry_every
        self.pending = 0  # flushes since the last carry propagation
        self.sp = _Specials()
        self.chunk_ops = 0  # chunk additions and carry steps, for the operation counts

    def add(self, x: float) -> None:
        if self.sp.take(x) or x == 0.0:
            return
        bits = f64_bits(x)
        ef = (bits >> 52) & 0x7FF
        m = bits & ((1 << 52) - 1)
        if ef:
            m |= 1 << 52
        if bits >> 63:
            m = -m
        e = self.large.get(ef)
        if e is None:
            self.large[ef] = [m, 1]
            e = self.large[ef]
        else:
            e[0] += m
            e[1] += 1
        assert abs(e[0]) <= INT64_MAX
        if e[1] >= self.large_limit:
            self._flush(ef, e[0])
            del self.large[ef]

    def _flush(self, ef: int, s: int) -> None:
        if s == 0:
            return
        p = ef - 1 if ef else 0
        c, sh = divmod(p, 32)
        a = abs(s) << sh  # < 2^94: three chunks
        sign = -1 if s < 0 else 1
        S = self.small
        for j in range(3):
            piece = (a >> (32 * j)) & MASK32
            if piece:
                v = S.get(c + j, 0) + sign * piece
                assert abs(v) <= INT64_MAX
                S[c + j] = v
                self.chunk_ops += 1
        self.pending += 1
        if self.pending >= self.carry_every:
            self._carry()

    def _carry(self) -> None:
        # Ripple from the lowest chunk: each chunk keeps its low 32 bits and
        # passes the rest (arithmetic shift) up. Past the top the carry settles
        # to 0 or -1 (the two's complement sign extension); a -1 becomes the top
        # chunk, moved down over any all-ones chunks so the form is minimal.
        S = self.small
        self.pending = 0
        if not S:
            return
        k, hi = min(S), max(S)
        carry = 0
        while k <= hi or carry not in (0, -1):
            v = S.pop(k, 0) + carry
            low = v & MASK32
            carry = v >> 32
            if low:
                S[k] = low
            k += 1
            self.chunk_ops += 1
        if carry == -1:
            top = k
            while S.get(top - 1) == MASK32:
                del S[top - 1]
                top -= 1
            S[top] = -1

    def _flush_all(self) -> None:
        for ef, (s, _cnt) in sorted(self.large.items()):
            self._flush(ef, s)
        self.large.clear()

    def merge(self, other: SuperAcc) -> None:
        o = other.copy()
        o._flush_all()
        o._carry()
        self._flush_all()
        self._carry()
        for k, v in o.small.items():
            self.small[k] = self.small.get(k, 0) + v
        self.chunk_ops += o.chunk_ops + len(o.small)
        self.pending = 2
        self._carry()
        self.sp.merge(other.sp)

    def copy(self) -> SuperAcc:
        c = SuperAcc(self.large_limit, self.carry_every)
        c.large = {k: list(v) for k, v in self.large.items()}
        c.small = dict(self.small)
        c.pending = self.pending
        c.sp.merge(self.sp)
        return c

    def to_int(self) -> int:
        n = 0
        for ef, (s, _c) in self.large.items():
            n += s << (ef - 1 if ef else 0)
        for k, v in self.small.items():
            n += v << (32 * k)
        return n

    def round_to_f64(self) -> float:
        s = self.sp.result()
        return s if s is not None else round_scaled_int(self.to_int())

    def canonical(self) -> tuple[tuple[int, int], ...]:
        c = self.copy()
        c._flush_all()
        c._carry()
        return tuple(sorted(c.small.items()))

    def state_bytes(self) -> int:
        # large entry: 2-byte key, 8-byte sum, 2-byte count; small chunk: 1-byte
        # index (0..68), 8-byte value; plus two 2-byte lengths and the flag byte
        return 12 * len(self.large) + 9 * len(self.small) + 5


ACCS = {"shewchuk": Shewchuk, "kulisch": Kulisch, "superacc": SuperAcc}


def run_acc(cls, xs):
    a = cls()
    for x in xs:
        a.add(x)
    return a


def exact_ref(xs: list[float]) -> float:
    fin = [x for x in xs if math.isfinite(x)]
    sp = _Specials()
    for x in xs:
        sp.take(x)
    s = sp.result()
    if s is not None:
        return s
    q = sum((Fraction(x) for x in fin), Fraction(0))
    try:
        return float(q) + 0.0
    except OverflowError:
        return math.inf if q > 0 else -math.inf


# --------------------------------------------------------------------------
# Generators (seeded)
# --------------------------------------------------------------------------


def gen_mixed(rng: np.random.Generator, n: int) -> list[float]:
    e = rng.integers(-60, 61, n)
    m = rng.uniform(1.0, 2.0, n)
    s = rng.choice([-1.0, 1.0], n)
    return [float(v) for v in np.ldexp(s * m, e)]


def gen_cancel(rng: np.random.Generator, n: int) -> list[float]:
    half = gen_mixed(rng, n // 2)
    big = [float(v) for v in rng.choice([-1, 1], 8) * np.ldexp(1.0, rng.integers(60, 200, 8))]
    small = [float(v) for v in np.ldexp(rng.uniform(-1, 1, 8), rng.integers(-200, -60, 8))]
    xs = half + [-v for v in half] + big + [-v for v in big] + small
    # perturb a few so the exact sum is tiny but rarely zero
    for _ in range(4):
        i = int(rng.integers(0, len(xs)))
        xs[i] = math.nextafter(xs[i], math.inf)
    order = rng.permutation(len(xs))
    return [xs[i] for i in order]


def gen_range(rng: np.random.Generator, n: int) -> list[float]:
    """Random bit patterns over the whole finite range, subnormals included,
    with the exponent field at most 2034 (below 2^1012) so 1000 terms cannot
    overflow."""
    out = []
    while len(out) < n:
        ef = int(rng.integers(0, 2035))
        frac = int(rng.integers(0, 1 << 52, dtype=np.uint64))
        sgn = int(rng.integers(0, 2))
        bits = (sgn << 63) | (ef << 52) | frac
        out.append(struct.unpack("<d", struct.pack("<Q", bits))[0])
    return out


def gen_subnormal(rng: np.random.Generator, n: int) -> list[float]:
    xs = [float(v) for v in np.ldexp(rng.integers(-(1 << 52), 1 << 52, n).astype(float), -1074)]
    xs += [2.2250738585072014e-308, -2.2250738585072014e-308 / 3]
    return xs


def gen_ties(rng: np.random.Generator, n: int) -> list[float]:
    """Exact sums at or a hair off a half-way point between two f64s."""
    a = float(rng.uniform(1, 2)) * 2.0 ** int(rng.integers(-500, 500))
    h = math.ulp(a) / 2
    k = int(rng.integers(0, 3))
    tiny = [0.0, h * 2.0**-60, -h * 2.0**-60][k]
    xs = [a, h, tiny] + gen_mixed(rng, n)
    mixed = xs[3:]
    xs += [-v for v in mixed]
    order = rng.permutation(len(xs))
    return [xs[i] for i in order]


def gen_overflow(rng: np.random.Generator, n: int) -> list[float]:
    """Near-overflow terms whose exact sum is finite, or just at the edge."""
    M = 1.7976931348623157e308
    xs = [M, M, -M] + [float(v) for v in rng.uniform(-1, 1, n) * 1e300]
    if rng.integers(0, 2):
        xs += [math.ulp(M) / 2]  # half-way above MAX: ties to even -> inf
    order = rng.permutation(len(xs))
    return [xs[i] for i in order]


GENS = {
    "mixed": gen_mixed,
    "cancelling": gen_cancel,
    "huge range": gen_range,
    "subnormal": gen_subnormal,
    "ties": gen_ties,
    "near overflow": gen_overflow,
}


# --------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------


def check_correctness(trials: int, rng: np.random.Generator) -> None:
    print("== 3. Rounded result against math.fsum and float(Fraction(exact sum)) ==")
    print(
        f"{'generator':<14} {'trials':>6}  " + "  ".join(f"{k + ' vs fsum/Frac':>24}" for k in ACCS)
    )
    tot = {k: [0, 0, 0] for k in ACCS}  # mismatches vs fsum, vs Fraction, trials compared
    for gname, gen in GENS.items():
        mism = {k: [0, 0] for k in ACCS}
        raised = {k: 0 for k in ACCS}
        fsum_raised = 0
        for _ in range(trials):
            n = int(rng.integers(1, 1001))
            xs = gen(rng, n)
            ref = exact_ref(xs)
            try:
                fs = math.fsum(xs) + 0.0
            except OverflowError:
                fs = None
                fsum_raised += 1
            for k, cls in ACCS.items():
                try:
                    r = run_acc(cls, xs).round_to_f64()
                except OverflowError:
                    raised[k] += 1
                    continue
                if fs is not None and not same_bits(r, fs):
                    mism[k][0] += 1
                if not same_bits(r, ref):
                    mism[k][1] += 1
                tot[k][2] += 1
        for k in ACCS:
            tot[k][0] += mism[k][0]
            tot[k][1] += mism[k][1]
        cells = []
        for k in ACCS:
            cell = f"{mism[k][0]}/{mism[k][1]}"
            if raised[k]:
                cell += f" (raised {raised[k]})"
            cells.append(f"{cell:>24}")
        note = f"  fsum raised {fsum_raised}" if fsum_raised else ""
        print(f"{gname:<14} {trials:>6}  " + "  ".join(cells) + note)
    # specials
    cases = [
        [1.0, math.inf],
        [1.0, -math.inf, 2.0],
        [math.inf, -math.inf],
        [math.nan, 1.0],
        [-0.0],
        [-0.0, -0.0],
        [0.0, -0.0],
        [],
        [5e-324, -5e-324],
        [5e-324] * 3,
    ]
    bad = 0
    for xs in cases:
        ref = exact_ref(xs)
        for cls in ACCS.values():
            r = run_acc(cls, xs).round_to_f64()
            if not (same_bits(r, ref) or (math.isnan(r) and math.isnan(ref))):
                bad += 1
                print("  special mismatch", cls.__name__, xs, r, ref)
    print(f"special values ({len(cases)} cases x 3): {bad} mismatches against the stated policy")
    for k in ACCS:
        t = tot[k]
        check(
            f"single sum, {k}: rounded result",
            "math.fsum where it returns; float(Fraction(exact sum))",
            f"{t[0]} / {t[1]} mismatches over {t[2]} sums",
            "0 (bit-identical)",
            t[0] == 0 and t[1] == 0,
        )
    check(
        "single sum, special values",
        "the stated flag policy",
        f"{bad} mismatches over {len(cases)} x 3",
        "0",
        bad == 0,
    )


def random_tree(items: list, combine, rng: random.Random):
    """Combine in a random order and a random grouping (a random binary tree)."""
    pool = list(items)
    rng.shuffle(pool)
    while len(pool) > 1:
        i, j = rng.sample(range(len(pool)), 2)
        v = combine(pool[i], pool[j])
        pool = [p for t, p in enumerate(pool) if t not in (i, j)] + [v]
    return pool[0]


def _merged(a, b):
    a.merge(b)
    return a


def random_merge(states: list, rng: random.Random):
    """Merge accumulator states by a random binary tree (in place)."""
    return random_tree(states, _merged, rng)


def float_random_merge(parts: list[float], rng: random.Random) -> float:
    return random_tree(parts, lambda a, b: a + b, rng)


def make_split(n: int, rng: random.Random) -> list[list[int]]:
    k = rng.randint(2, 16)
    idx = list(range(n))
    rng.shuffle(idx)  # shards in no time order
    cuts = sorted(rng.sample(range(1, n), k - 1))
    bounds = [0] + cuts + [n]
    return [idx[bounds[t] : bounds[t + 1]] for t in range(k)]


def check_shards(n_sets: int, n_splits: int, seed: int) -> None:
    print()
    print("== 4. Shard invariance: 1000 values, 2..16 shards, random merge order and grouping ==")
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    gens = [gen_mixed, gen_cancel, gen_range]
    diff_result = {k: 0 for k in ACCS}
    diff_state = {k: 0 for k in ACCS}
    diff_raw = {k: 0 for k in ACCS}
    diff_frac = {k: 0 for k in ACCS}
    total = 0
    float_split_diff = 0
    float_set_diff = 0
    for t in range(n_sets):
        xs = gens[t % 3](nrng, 1000)[:1000]
        base = {k: run_acc(cls, xs) for k, cls in ACCS.items()}
        base_r = {k: f64_bits(a.round_to_f64()) for k, a in base.items()}
        base_c = {k: a.canonical() for k, a in base.items()}
        base_raw = {"shewchuk": tuple(base["shewchuk"].p)}
        frac = f64_bits(exact_ref(xs))
        seq = 0.0
        for x in xs:
            seq += x
        any_float = False
        for _ in range(n_splits):
            total += 1
            split = make_split(len(xs), rng)
            for k, cls in ACCS.items():
                if k == "superacc":
                    # vary the flush and carry intervals per shard: they must not matter
                    states = []
                    for sh in split:
                        a = SuperAcc(rng.choice([1, 3, 7, 1024]), rng.choice([1, 5, (1 << 31) - 1]))
                        for i in sh:
                            a.add(xs[i])
                        states.append(a)
                else:
                    states = [run_acc(cls, [xs[i] for i in sh]) for sh in split]
                m = random_merge(states, rng)
                rb = f64_bits(m.round_to_f64())
                if rb != base_r[k]:
                    diff_result[k] += 1
                if rb != frac:
                    diff_frac[k] += 1
                if m.canonical() != base_c[k]:
                    diff_state[k] += 1
                if k == "shewchuk" and tuple(m.p) != base_raw["shewchuk"]:
                    diff_raw[k] += 1
            parts = []
            for sh in split:
                s = 0.0
                for i in sh:
                    s += xs[i]
                parts.append(s)
            fv = float_random_merge(parts, rng)
            if f64_bits(fv) != f64_bits(seq):
                float_split_diff += 1
                any_float = True
        float_set_diff += any_float
    print(f"{n_sets} sets x {n_splits} splits = {total} sharded merges per accumulator")
    for k in ACCS:
        extra = f"; raw partials differ in {diff_raw[k]}" if k == "shewchuk" else ""
        print(
            f"  {k:<9} rounded result differs: {diff_result[k]}/{total}"
            f" (from float(Fraction): {diff_frac[k]}/{total});"
            f" canonical state differs: {diff_state[k]}/{total}{extra}"
        )
        check(
            f"shard merge, no decay, {k}: rounded result",
            "float(Fraction(exact sum of the set))",
            f"{diff_frac[k]} / {total} splits differ",
            "0 (bit-identical)",
            diff_frac[k] == 0 and diff_result[k] == 0,
        )
        check(
            f"shard merge, no decay, {k}: canonical state",
            "the unsharded run's canonical state",
            f"{diff_state[k]} / {total} splits differ",
            "0 (byte-identical)",
            diff_state[k] == 0,
        )
    print(
        f"  plain float sum: differs from the sequential sum in {float_split_diff}/{total} splits"
        f" ({float_set_diff}/{n_sets} sets with at least one)"
    )


def report_state(seed: int) -> None:
    print()
    print("== 5. State bytes ==")
    nrng = np.random.default_rng(seed)
    # Shewchuk worst case: one-bit components 54 bits apart over the whole range
    adv = [2.0**e for e in range(1023, -1075, -54)]
    w = run_acc(Shewchuk, adv)
    print(
        f"  shewchuk: 8 bytes x components + 2; canonical worst case"
        f" {len(w.canonical())} components ({8 * len(w.canonical()) + 2} B),"
        f" reached by {len(adv)} powers of two 54 bits apart"
    )
    worst_raw = worst_can = 0
    for _ in range(20):
        a = run_acc(Shewchuk, gen_range(nrng, 1000))
        worst_raw = max(worst_raw, len(a.p))
        worst_can = max(worst_can, len(a.canonical()))
    print(
        f"    20 full-range sets of 1000: raw expansion at most {worst_raw} components"
        f" ({8 * worst_raw + 2} B); canonical at most {worst_can} ({8 * worst_can + 2} B)"
    )
    can_max = max(len(w.canonical()), worst_can)
    check(
        "state bytes, shewchuk canonical",
        "8 x components + 2, components <= 2098 / 53 + 1 = 40",
        f"at most {8 * can_max + 2} B ({can_max} components)",
        "<= 322 B",
        8 * can_max + 2 <= 322,
    )
    sup_max = 0
    for gname in ("mixed", "cancelling", "huge range"):
        xs = GENS[gname](nrng, 1000)
        a = run_acc(Shewchuk, xs)
        b = run_acc(SuperAcc, xs)
        bc = b.copy()
        bc._flush_all()
        bc._carry()
        sup_max = max(sup_max, bc.state_bytes())
        print(
            f"    {gname:<10} 1000 values: shewchuk {len(a.p)} components = {a.state_bytes()} B;"
            f" superacc {len(b.large)} large + {len(b.small)} small live = {b.state_bytes()} B"
            f" (canonical {len(bc.small)} chunks = {bc.state_bytes()} B)"
        )
    print(
        f"  kulisch: {K_LIMBS} limbs x 8 = {8 * K_LIMBS} B + 1 flag byte"
        f" = {Kulisch.state_bytes()} B, fixed"
    )
    print(
        "    (exact dot products, products down to 2^-2148: 4196 + 64 + 1 bits -> 67 limbs = 536 B)"
    )
    # superacc worst case: every exponent key live, every chunk live
    print(
        "  superacc: 12 B x live large entries + 9 B x live small chunks + 5 B;"
        f" worst case 2048 x 12 + 69 x 9 + 5 = {2048 * 12 + 69 * 9 + 5} B live,"
        f" canonical (large flushed) at most 69 x 9 + 5 = {69 * 9 + 5} B"
    )
    check(
        "state bytes, kulisch",
        "34 limbs x 8 + 1 flag byte",
        f"{Kulisch.state_bytes()} B",
        "== 273 B, fixed",
        Kulisch.state_bytes() == 273,
    )
    check(
        "state bytes, superacc canonical",
        "9 x chunks + 5, chunks <= 69",
        f"at most {sup_max} B on these data",
        "<= 626 B",
        sup_max <= 626,
    )


def run_accumulators(quick: bool) -> None:
    print("######## accumulators (part 1) ########")
    rng = np.random.default_rng(SEED)
    check_correctness(40 if quick else 200, rng)
    check_shards(6 if quick else 24, 5 if quick else 12, SEED)
    report_state(SEED)


# ==========================================================================
# merge (part 2): raw weighted sums of a Gram, merged across shards
# ==========================================================================
#
# 1. Without decay, merging shard states is adding their sums. The merged
#    Gram (weight, first moments, second moments of k features) is compared
#    with the unsharded state over random shardings: Kulisch against plain
#    float sums, and the mean form (weighted Welford, merged by Chan's
#    formula) for contrast.
# 2. With decay, a shard's sums are taken at its own clock T. Merging two
#    needs S_A * lam^(T_B - T_A) + S_B. The study measures what that rounded
#    power costs against the exact sum of every row in one common clock, for
#    contiguous shards (general and power-of-two offsets) and interleaved ones.

N_ROWS = 500

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def ordered(x: float) -> int:
    """The f64's bits as an integer ordered like the reals (for ulp distances)."""
    b = f64_bits(x)
    return -(b & ((1 << 63) - 1)) if b >> 63 else b


def ulps(a: float, b: float) -> int:
    return abs(ordered(a) - ordered(b))


def kulisch_from_int(n: int) -> Kulisch:
    acc = Kulisch()
    n %= 1 << K_BITS
    acc.limbs = [(n >> (64 * i)) & MASK64 for i in range(K_LIMBS)]
    return acc


def scale_exact_then_round(acc: Kulisch, f: float) -> Kulisch:
    """N * f rounded once to the 2^-1074 grid (ties to even): the best a
    register can do when multiplied by a float. f = M / 2^E exactly."""
    num, den = f.as_integer_ratio()  # den is a power of two
    n = acc.to_int() * num
    q, r = divmod(n, den)
    if 2 * r > den or (2 * r == den and q & 1):
        q += 1
    out = kulisch_from_int(q)
    out.sp = acc.sp
    return out


SPLIT_POW = False


def pow_lam(d: int, h: int) -> float:
    """lam^d for lam = 2^(-1/h): one libm power from the exact integer offset.

    With SPLIT_POW, 2^-(d div h) * fl(2^(-(d mod h) / h)): whole half-lives
    are an exact power of two, so lam^(d + n h) = lam^d * 2^-n to the bit.
    """
    if SPLIT_POW:
        q, r = divmod(d, h)
        return math.ldexp(math.pow(2.0, -r / h), -q)
    return math.pow(2.0, -d / h)


def gen_rows(rng: np.random.Generator, n: int, k: int) -> tuple[list[float], list[list[float]]]:
    """Weights in [0.5, 2) and features at mixed levels and scales (some offset)."""
    w = rng.uniform(0.5, 2.0, n).tolist()
    level = rng.choice([0.0, 3.0, 1e3], k)
    scale = 10.0 ** rng.uniform(-2, 2, k)
    x = (level + scale * rng.standard_normal((n, k))).tolist()
    return w, x


def row_terms(w: float, x: list[float]) -> list[float]:
    """One row's state terms, each rounded once per row (deterministic):
    W = w; S1_i = w x_i; S2_ij = (w x_i) x_j for i <= j."""
    k = len(x)
    wx = [w * xi for xi in x]
    out = [w] + wx
    for i in range(k):
        for j in range(i, k):
            out.append(wx[i] * x[j])
    return out


def field_names(k: int) -> list[str]:
    return (
        ["W"]
        + [f"S1_{i}" for i in range(k)]
        + [f"S2_{i}{j}" for i in range(k) for j in range(i, k)]
    )


def make_row_split(n: int, rng: random.Random, contiguous: bool) -> list[list[int]]:
    s = rng.randint(2, 16)
    idx = list(range(n))
    if not contiguous:
        rng.shuffle(idx)
    cuts = sorted(rng.sample(range(1, n), s - 1))
    b = [0] + cuts + [n]
    return [sorted(idx[b[t] : b[t + 1]]) for t in range(s)]


# --------------------------------------------------------------------------
# merge (part 2), 1: no decay
# --------------------------------------------------------------------------


def kulisch_state(terms: list[list[float]], rows: list[int]) -> list[Kulisch]:
    accs = [Kulisch() for _ in terms[0]]
    for r in rows:
        for a, t in zip(accs, terms[r], strict=True):
            a.add(t)
    return accs


def kulisch_merge(a: list[Kulisch], b: list[Kulisch]) -> list[Kulisch]:
    for x, y in zip(a, b, strict=True):
        x.merge(y)
    return a


def float_state(terms: list[list[float]], rows: list[int]) -> list[float]:
    s = [0.0] * len(terms[0])
    for r in rows:
        s = [si + ti for si, ti in zip(s, terms[r], strict=True)]
    return s


def welford(w: list[float], x: list[list[float]], rows: list[int]):
    """Weighted Welford in rheon's mean form: (W, mean m, normalized co-moments C)."""
    k = len(x[0])
    W, m, C = 0.0, [0.0] * k, [[0.0] * k for _ in range(k)]
    for r in rows:
        wn = W + w[r]
        a, b = W / wn, w[r] / wn
        d = [x[r][i] - m[i] for i in range(k)]
        for i in range(k):
            for j in range(k):
                C[i][j] = a * (C[i][j] + b * d[i] * d[j])
        m = [m[i] + b * d[i] for i in range(k)]
        W = wn
    return W, m, C


def chan(sa, sb):
    """Chan et al.'s pairwise merge of two mean-form states."""
    (Wa, ma, Ca), (Wb, mb, Cb) = sa, sb
    k = len(ma)
    W = Wa + Wb
    d = [mb[i] - ma[i] for i in range(k)]
    m = [ma[i] + d[i] * (Wb / W) for i in range(k)]
    C = [
        [(Wa * Ca[i][j] + Wb * Cb[i][j]) / W + d[i] * d[j] * (Wa * Wb / (W * W)) for j in range(k)]
        for i in range(k)
    ]
    return W, m, C


def flat_mean_form(s) -> list[float]:
    W, m, C = s
    k = len(m)
    return [W] + m + [C[i][j] for i in range(k) for j in range(i, k)]


def check_no_decay(n_sets: int, n_splits: int) -> None:
    print("== 2. Merge without decay: 500 rows, 2..16 shards (rows shuffled), random merge tree ==")
    for k in (2, 5):
        rng = np.random.default_rng(SEED + k)
        prng = random.Random(SEED * 10 + k)
        n_fields = 1 + k + k * (k + 1) // 2
        tot = {"kulisch": [0, 0, 0], "float": [0, 0, 0], "meanform": [0, 0, 0]}
        frac_bad = 0  # merged Kulisch fields whose rounding differs from the Fraction oracle
        worst = {"float": 0, "meanform": 0}
        splits = 0
        for _ in range(n_sets):
            w, x = gen_rows(rng, N_ROWS, k)
            terms = [row_terms(w[r], x[r]) for r in range(N_ROWS)]
            allrows = list(range(N_ROWS))
            ref_k = [a.canonical() for a in kulisch_state(terms, allrows)]
            ref_q = [
                f64_bits(float(sum((Fraction(t[f]) for t in terms), Fraction(0))))
                for f in range(n_fields)
            ]
            ref_f = float_state(terms, allrows)
            ref_m = flat_mean_form(welford(w, x, allrows))
            for _ in range(n_splits):
                splits += 1
                parts = make_row_split(N_ROWS, prng, contiguous=False)
                mk = random_tree([kulisch_state(terms, p) for p in parts], kulisch_merge, prng)
                diff = sum(a.canonical() != b for a, b in zip(mk, ref_k, strict=True))
                tot["kulisch"][0] += diff > 0
                tot["kulisch"][1] += diff
                rounded = [f64_bits(a.round_to_f64()) for a in mk]
                frac_bad += sum(a != q for a, q in zip(rounded, ref_q, strict=True))
                mf = random_tree(
                    [float_state(terms, p) for p in parts],
                    lambda a, b: [u + v for u, v in zip(a, b, strict=True)],
                    prng,
                )
                diff = [ulps(a, b) for a, b in zip(mf, ref_f, strict=True)]
                tot["float"][0] += any(diff)
                tot["float"][1] += sum(d > 0 for d in diff)
                worst["float"] = max(worst["float"], max(diff))
                mm = flat_mean_form(random_tree([welford(w, x, p) for p in parts], chan, prng))
                diff = [ulps(a, b) for a, b in zip(mm, ref_m, strict=True)]
                tot["meanform"][0] += any(diff)
                tot["meanform"][1] += sum(d > 0 for d in diff)
                worst["meanform"] = max(worst["meanform"], max(diff))
        ent = splits * n_fields
        print(
            f" k = {k}: {n_fields} fields ({field_names(k)[:4]} ...), {splits} splits,"
            f" {ent} field values"
        )
        print(
            f"   Kulisch sums       : splits differing {tot['kulisch'][0]:4d} / {splits};"
            f" fields differing {tot['kulisch'][1]:5d} / {ent} (limbs compared);"
            f" rounded fields off float(Fraction(exact sum)): {frac_bad} / {ent}"
        )
        check(
            f"raw-sum merge, no decay, k = {k}: every field",
            "float(Fraction(exact sum of the per-row terms)); limbs vs the unsharded run",
            f"{frac_bad} / {ent} fields off the oracle; {tot['kulisch'][0]} / {splits} splits'"
            " limbs differ",
            "0 (bit-identical; limbs byte-identical)",
            frac_bad == 0 and tot["kulisch"][0] == 0,
        )
        print(
            f"   plain float sums   : splits differing {tot['float'][0]:4d} / {splits};"
            f" fields differing {tot['float'][1]:5d} / {ent}; worst {worst['float']} ulps"
        )
        print(
            f"   mean form + Chan   : splits differing {tot['meanform'][0]:4d} / {splits};"
            f" fields differing {tot['meanform'][1]:5d} / {ent}; worst {worst['meanform']} ulps"
        )


def check_centering() -> None:
    """Raw sums lose the covariance under an offset in floats; exact sums do not."""
    print()
    print("== 2b. Centred covariance read from raw sums: offset 1e8, k = 2, 500 rows ==")
    rng = np.random.default_rng(SEED)
    w = rng.uniform(0.5, 2.0, N_ROWS).tolist()
    x = (1e8 + rng.standard_normal((N_ROWS, 2))).tolist()
    terms = [row_terms(w[r], x[r]) for r in range(N_ROWS)]
    rows = list(range(N_ROWS))
    # Truth: the weighted covariance of the data themselves, in rationals.
    fw = sum(Fraction(v) for v in w)
    fs = [sum(Fraction(w[r]) * Fraction(x[r][i]) for r in rows) for i in range(2)]
    fq = sum(Fraction(w[r]) * Fraction(x[r][0]) * Fraction(x[r][1]) for r in rows)
    truth = float(fq / fw - fs[0] * fs[1] / (fw * fw))

    def read(n: list[int]) -> float:
        W, S0, S1, S01 = (Fraction(v, 1 << 1074) for v in n)
        return float(S01 / W - S0 * S1 / (W * W))

    # Exact sums of the rounded per-row products (34 limbs).
    ks = kulisch_state(terms, rows)
    from_rounded = read([ks[i].to_int() for i in (0, 1, 2, 4)])
    # Exact sums of exact products: w x_i split by FMA into hi + lo, each
    # times x_j split again, four f64s per product, all into the register.
    ex = [Kulisch() for _ in range(4)]
    for r in rows:
        ex[0].add(w[r])
        wx = []
        for i in range(2):
            hi = w[r] * x[r][i]
            lo = math.fma(w[r], x[r][i], -hi)
            ex[1 + i].add(hi)
            ex[1 + i].add(lo)
            wx.append((hi, lo))
        for part in wx[0]:
            p = part * x[r][1]
            ex[3].add(p)
            ex[3].add(math.fma(part, x[r][1], -p))
    from_exact = read([a.to_int() for a in ex])
    f = float_state(terms, rows)
    from_float = f[4] / f[0] - (f[1] / f[0]) * (f[2] / f[0])
    _, _, C = welford(w, x, rows)
    rel = lambda v: abs(v - truth) / abs(truth)  # noqa: E731
    print(f"   truth (Fraction over the data)                    : {truth:.17g}")
    print(
        "   exact sums of exact products (FMA split), 1 rounding:"
        f" {from_exact:.17g}  ({ulps(from_exact, truth)} ulps)"
    )
    print(
        "   exact sums of per-row rounded products            :"
        f" {from_rounded:.17g}  (rel. error {rel(from_rounded):.1e})"
    )
    print(
        "   plain float raw sums                              :"
        f" {from_float:.17g}  (rel. error {rel(from_float):.1e})"
    )
    print(
        "   rheon's mean form (weighted Welford, floats)      :"
        f" {C[0][1]:.17g}  (rel. error {rel(C[0][1]):.1e})"
    )


# --------------------------------------------------------------------------
# merge (part 2), 2: decay
# --------------------------------------------------------------------------


def gen_clock(rng: random.Random, n: int, h: int, aligned: bool) -> list[int]:
    """Row times on an integer clock: gaps of 1..3 ticks, or multiples of h."""
    t, out = 0, []
    for _ in range(n):
        t += h * rng.randint(1, 2) if aligned else rng.randint(1, 3)
        out.append(t)
    return out


def decayed_kulisch(terms, ts, rows, T, h) -> list[Kulisch]:
    """Exact sum of fl(lam^(T - t) * term) over the rows: the rows in clock T."""
    accs = [Kulisch() for _ in terms[0]]
    for r in rows:
        p = pow_lam(T - ts[r], h)
        for a, t in zip(accs, terms[r], strict=True):
            a.add(p * t)
    return accs


def decayed_float(terms, ts, rows, T, h) -> list[float]:
    s = [0.0] * len(terms[0])
    for r in rows:
        p = pow_lam(T - ts[r], h)
        s = [si + p * ti for si, ti in zip(s, terms[r], strict=True)]
    return s


def streaming_float(terms, ts, h) -> list[float]:
    """rheon's own recursion: S <- lam^(dt) S + term, row by row."""
    s, last = [0.0] * len(terms[0]), ts[0]
    for r in range(len(ts)):
        p = pow_lam(ts[r] - last, h)
        s = [p * si + ti for si, ti in zip(s, terms[r], strict=True)]
        last = ts[r]
    return s


# (k, case) -> (expectation, [splits differing, fields differing, worst ulps, splits])
DECAY_RESULTS: dict = {}
ORACLE_GAP = [0, 0]  # unsharded Kulisch decayed sums off the Fraction oracle, fields compared
DECAY_TOL_ULPS = 13  # the measured worst with the accumulator across general offsets


def check_decay(n_sets: int, n_splits: int) -> None:
    print()
    print(
        "== 3-4. Merge with decay: lam = 2^(-1/h), integer clock, 500 rows, k = 2 and 5,"
        " 2..16 shards =="
    )
    print(
        "   oracle: float(Fraction(exact sum of every row's fl(fl(lam^(T_end - t)) * term))),"
        " one common clock; the unsharded Kulisch sum is checked against it"
    )
    # Composition of rounded powers.
    prng = random.Random(SEED)
    comp_bad = comp_exact_bad = 0
    trials = 20000
    h = 50
    for _ in range(trials):
        a, b = prng.randint(1, 3000), prng.randint(1, 3000)
        pa, pb, pab = pow_lam(a, h), pow_lam(b, h), pow_lam(a + b, h)
        comp_bad += (pa * pb) != pab
        comp_exact_bad += Fraction(pa) * Fraction(pb) != Fraction(pab)
    print(
        "   rounded power: fl(fl(lam^a) * fl(lam^b)) != fl(lam^(a+b))"
        f" in {comp_bad} / {trials} pairs (h = {h});"
        f" the exact product != fl(lam^(a+b)) in {comp_exact_bad} / {trials}"
    )

    cases = [
        ("common clock T_end, interleaved shards (add only)", False, "free", "common", "0"),
        (
            "contiguous, own clocks, register rounded to f64 then * lam^dT",
            True,
            "free",
            "round",
            "tol",
        ),
        (
            "contiguous, own clocks, register * lam^dT rounded once to the grid",
            True,
            "free",
            "exact",
            "tol",
        ),
        ("contiguous, own clocks, plain float sums * lam^dT", True, "free", "float", None),
        (
            "contiguous, every row time a multiple of h (all powers 2^-n)",
            True,
            "rows_h",
            "exact",
            "0",
        ),
        (
            "contiguous, shard clocks on the h grid (lam^dT = 2^-n), rows free",
            True,
            "shards_h",
            "exact",
            "tol",
        ),
        (
            "interleaved, own clocks, register * lam^dT rounded once to the grid",
            False,
            "free",
            "exact",
            "tol",
        ),
        (
            "interleaved, shard clocks on the h grid (lam^dT = 2^-n), rows free",
            False,
            "shards_h",
            "exact",
            "tol",
        ),
        ("interleaved, own clocks, plain float sums * lam^dT", False, "free", "float", None),
        (
            "contiguous, shard clocks on the h grid, power split by whole half-lives",
            True,
            "shards_h",
            "exact+split",
            "0",
        ),
        (
            "interleaved, shard clocks on the h grid, power split by whole half-lives",
            False,
            "shards_h",
            "exact+split",
            "0",
        ),
    ]
    for k in (2, 5):
        n_fields = 1 + k + k * (k + 1) // 2
        print(f" k = {k} ({n_fields} fields):")
        results = {c[0]: [0, 0, 0, 0] for c in cases}
        stream = [0, 0, 0]
        for ci, (name, contiguous, clock, mode, _expect) in enumerate(cases):
            global SPLIT_POW
            SPLIT_POW = mode.endswith("+split")
            mode = mode.removesuffix("+split")
            rng = np.random.default_rng(SEED + 100 * k)
            crng = random.Random(SEED + k)
            srng = random.Random(SEED + 7 * k + ci)
            for _ in range(n_sets):
                w, x = gen_rows(rng, N_ROWS, k)
                terms = [row_terms(w[r], x[r]) for r in range(N_ROWS)]
                ts = gen_clock(crng, N_ROWS, h, clock == "rows_h")
                T_end = ts[-1]
                ref = [
                    a.round_to_f64() for a in decayed_kulisch(terms, ts, range(N_ROWS), T_end, h)
                ]
                pw = [pow_lam(T_end - t, h) for t in ts]
                ref_q = [
                    float(sum((Fraction(pw[r] * terms[r][f]) for r in range(N_ROWS)), Fraction(0)))
                    for f in range(n_fields)
                ]
                ORACLE_GAP[0] += sum(not same_bits(a, b) for a, b in zip(ref, ref_q, strict=True))
                ORACLE_GAP[1] += n_fields
                ref = ref_q
                if ci == 0:
                    sf = streaming_float(terms, ts, h)
                    d = [ulps(a, b) for a, b in zip(sf, ref, strict=True)]
                    stream[0] += any(d)
                    stream[1] += sum(v > 0 for v in d)
                    stream[2] = max(stream[2], max(d))
                for _ in range(n_splits):
                    parts = make_row_split(N_ROWS, srng, contiguous)
                    if mode == "common":
                        st = [decayed_kulisch(terms, ts, p, T_end, h) for p in parts]
                        merged = random_tree(st, kulisch_merge, srng)
                        out = [a.round_to_f64() for a in merged]
                    else:
                        # Each shard in its own clock: its last row's time, or
                        # the first point at or after it on a grid of h back
                        # from T_end, where every offset's power is 2^-n.
                        def own(p, T_end=T_end, clock=clock, ts=ts):
                            last = ts[p[-1]]
                            return (
                                T_end - h * ((T_end - last) // h) if clock == "shards_h" else last
                            )

                        if mode == "float":
                            st = [(own(p), decayed_float(terms, ts, p, own(p), h)) for p in parts]

                            def comb(a, b):
                                (Ta, Sa), (Tb, Sb) = sorted([a, b], key=lambda s: s[0])
                                f = pow_lam(Tb - Ta, h)
                                return Tb, [f * u + v for u, v in zip(Sa, Sb, strict=True)]
                        else:
                            st = [(own(p), decayed_kulisch(terms, ts, p, own(p), h)) for p in parts]

                            def comb(a, b, mode=mode):
                                (Ta, Sa), (Tb, Sb) = sorted([a, b], key=lambda s: s[0])
                                f = pow_lam(Tb - Ta, h)
                                for ua, ub in zip(Sa, Sb, strict=True):
                                    if mode == "round":
                                        ub.add(f * ua.round_to_f64())
                                    else:
                                        ub.merge(scale_exact_then_round(ua, f))
                                return Tb, Sb

                        Tm, merged = random_tree(st, comb, srng)
                        assert Tm == T_end
                        out = [a if isinstance(a, float) else a.round_to_f64() for a in merged]
                    d = [ulps(a, b) for a, b in zip(out, ref, strict=True)]
                    r = results[name]
                    r[0] += any(d)
                    r[1] += sum(v > 0 for v in d)
                    r[2] = max(r[2], max(d))
                    r[3] += 1
            r = results[name]
            print(
                f"   {name:<72} splits differing {r[0]:3d} / {r[3]};"
                f" fields {r[1]:4d} / {r[3] * n_fields}; worst {r[2]} ulps"
            )
        SPLIT_POW = False
        for name, *_rest, expect in cases:
            DECAY_RESULTS[(k, name)] = (expect, results[name])
        print(
            f"   {'unsharded streaming recursion S <- lam^dt S + term (floats)':<72}"
            f" sets differing {stream[0]:3d} / {n_sets};"
            f" fields {stream[1]:4d} / {n_sets * n_fields}; worst {stream[2]} ulps"
        )


def decay_checks() -> None:
    """The decayed-merge claims against the Fraction oracle on the common clock."""
    check(
        "decayed sums, unsharded Kulisch",
        "float(Fraction(exact common-clock sum))",
        f"{ORACLE_GAP[0]} / {ORACLE_GAP[1]} fields differ",
        "0 (bit-identical)",
        ORACLE_GAP[0] == 0,
    )
    exact = [(k, n, r) for (k, n), (e, r) in DECAY_RESULTS.items() if e == "0"]
    tol = [(k, n, r) for (k, n), (e, r) in DECAY_RESULTS.items() if e == "tol"]
    bad = sum(r[0] for _, _, r in exact)
    n_ex = sum(r[3] for _, _, r in exact)
    check(
        "decayed merge, one common clock or whole-half-life offsets (accumulator)",
        "float(Fraction(exact common-clock sum))",
        f"{bad} / {n_ex} splits differ ({len(exact)} cases, k = 2 and 5)",
        "0 (bit-identical)",
        bad == 0,
    )
    worst = max(r[2] for _, _, r in tol)
    n_tol = sum(r[3] for _, _, r in tol)
    off = sum(r[0] for _, _, r in tol)
    per_k = {k: max(r[2] for kk, _, r in tol if kk == k) for k in (2, 5)}
    check(
        "decayed merge, general clock offsets (accumulator)",
        "float(Fraction(exact common-clock sum))",
        f"worst {worst} ulps (k = 2: {per_k[2]}, k = 5: {per_k[5]}); {off} / {n_tol} splits"
        f" off, {len(tol)} cases",
        f"<= {DECAY_TOL_ULPS} ulps",
        worst <= DECAY_TOL_ULPS,
    )


def run_merge(quick: bool) -> None:
    print("######## merge (part 2) ########")
    check_no_decay(3 if quick else 8, 4 if quick else 12)
    check_centering()
    check_decay(2 if quick else 4, 3 if quick else 8)
    decay_checks()


# ==========================================================================
# costs (part 3): operations and bytes, counted on the data, not timed
# ==========================================================================
#
# Units. "fp op": one f64 add, subtract, multiply or compare. "word op": one
# integer operation on a 64-bit word (shift, mask, add, compare). "limb op":
# one add or subtract with carry (or borrow), negation step or nonzero test of
# a 64-bit limb (Kulisch) or of a 64-bit chunk holding 32 payload bits
# (superaccumulator). Fixed per-add word ops are read off the add path; the
# variable parts (carry ripple, components, flushes, carry steps, scans) are
# counted while running the study's own data.


def round_digits(d: list[int], w: int) -> tuple[float, int]:
    """Round N * 2^-1074 to f64, ties to even, digit by digit; count digit ops.

    N is two's complement in ``len(d)`` digits of w bits, lowest first. Steps:
    the sign test; a negation if negative (one op per digit); a scan down for
    the top nonzero digit; the 53-bit window (at most 2 digits for w = 64, 3
    for w = 32); the round bit; and, only on a round bit of 1 with an even
    window, a sticky scan down to the first nonzero digit.
    """
    mask = (1 << w) - 1
    ops = 1
    sign = 1.0
    if d[-1] >> (w - 1):
        carry, out = 1, []
        for x in d:
            t = (x ^ mask) + carry
            out.append(t & mask)
            carry = t >> w
        ops += len(d)
        d, sign = out, -1.0
    top = len(d) - 1
    while top >= 0 and d[top] == 0:
        top -= 1
        ops += 1
    if top < 0:
        return 0.0, ops
    ops += 1
    length = top * w + d[top].bit_length()
    if length <= 53:
        n = 0
        for i in range(top, -1, -1):
            n = (n << w) | d[i]
        ops += top
        return sign * math.ldexp(n, -1074), ops
    shift = length - 53
    lo = shift // w
    n = 0
    for i in range(top, lo - 1, -1):
        n = (n << w) | d[i]
    ops += top - lo
    q = n >> (shift - lo * w)
    rpos = shift - 1
    ri, rb = divmod(rpos, w)
    if ri < lo:
        ops += 1
    if (d[ri] >> rb) & 1:
        if not q & 1:
            sticky = d[ri] & ((1 << rb) - 1) != 0
            i = ri - 1
            while not sticky and i >= 0:
                sticky = d[i] != 0
                ops += 1
                i -= 1
        else:
            sticky = True
        if sticky:
            q += 1
            if q == 1 << 53:
                q >>= 1
                shift += 1
    try:
        r = math.ldexp(q, shift - 1074)
    except OverflowError:
        r = math.inf
    return sign * r, ops


def superacc_digits(a: SuperAcc) -> list[int]:
    """The canonical small level as dense two's complement 32-bit digits."""
    S = a.small
    if not S:
        return [0]
    top = max(S)
    d = [S.get(i, 0) & MASK32 for i in range(top + 1)]
    if S[top] != -1:
        d.append(0)  # a zero sign digit for a positive sum
    return d


class Stat:
    def __init__(self) -> None:
        self.n = 0
        self.sum = 0
        self.max = 0

    def add(self, v: int) -> None:
        self.n += 1
        self.sum += v
        self.max = max(self.max, v)

    def __str__(self) -> str:
        return f"mean {self.sum / max(self.n, 1):.2f}, max {self.max}"


# Fixed word ops per add, read off each add path (decode, index, split):
# Kulisch: exponent (shift, mask), fraction (mask), hidden bit (test, OR),
#   position (subtract), sign (shift) = 7; limb index and offset (shift, mask)
#   = 2; the shifted mantissa's two words (two shifts) = 2: 11.
# Superacc: exponent (shift, mask), fraction (mask), hidden bit (test, OR),
#   sign and negate (2) = 7; the entry's sum (add), count (add), limit
#   (compare) = 3: 10. Each flush: position (subtract), chunk and offset
#   (shift, mask), shift the sum, three pieces (shift, mask each) = 10 word
#   ops plus up to 3 limb ops; it happens once per large_limit adds to an
#   entry.
# Shewchuk: per component, a two-sum (3 fp add/sub) plus a magnitude compare
#   (2 abs, 1 compare) and a zero test: 7 fp ops; the append is a store.
KULISCH_ADD_WORD = 11
SUPERACC_ADD_WORD = 10
SUPERACC_FLUSH_WORD = 10
SHEWCHUK_FP_PER_TWO_SUM = 7


def run_costs(quick: bool) -> None:
    print("######## costs (part 3): operations and bytes, counted, not timed ########")
    rng = np.random.default_rng(SEED + 3)
    prng = random.Random(SEED + 3)
    trials = 8 if quick else 30
    add = {k: Stat() for k in ACCS}
    rnd = {k: Stat() for k in ACCS}
    canon = {"shewchuk": Stat(), "superacc": Stat()}
    comps = Stat()
    chunks_live = Stat()
    entries_live = Stat()
    flush_adds = [0, 0]  # flushes, adds
    round_bad = 0
    n_sums = 0
    for gname, gen in GENS.items():
        for _ in range(trials):
            xs = gen(rng, int(rng.integers(1, 1001)))
            # the expansion raises on near-overflow sums, as fsum does
            xs_sh = None if gname == "near overflow" else xs
            ref = exact_ref(xs)
            n_sums += 1
            # Kulisch: 2 limb ops per add plus the carry ripple
            a = Kulisch()
            for x in xs:
                r0 = a.ripple
                a.add(x)
                if math.isfinite(x) and x != 0.0:
                    add["kulisch"].add(2 + a.ripple - r0)
            v, ops = round_digits(list(a.limbs), 64)
            rnd["kulisch"].add(ops)
            round_bad += not same_bits(v, a.round_to_f64()) or not same_bits(v, ref)
            # superaccumulator: chunk ops from flushes and carries, amortized
            s = SuperAcc()
            for x in xs:
                c0 = s.chunk_ops
                s.add(x)
                add["superacc"].add(s.chunk_ops - c0)
            entries_live.add(len(s.large))
            c = s.copy()
            c._flush_all()
            c._carry()
            flush_adds[0] += len(s.large)
            flush_adds[1] += len(xs)
            canon["superacc"].add(c.chunk_ops)
            chunks_live.add(len(c.small))
            v, ops = round_digits(superacc_digits(c), 32)
            rnd["superacc"].add(c.chunk_ops + ops)
            round_bad += not same_bits(v, s.round_to_f64()) or not same_bits(v, ref)
            # Shewchuk: two-sums per add; fsum's rounding steps; canonical form
            if xs_sh is not None:
                e = Shewchuk()
                for x in xs_sh:
                    t0 = e.two_sums
                    e.add(x)
                    add["shewchuk"].add(e.two_sums - t0)
                comps.add(len(e.p))
                v, steps = fsum_round(e.p)
                rnd["shewchuk"].add(steps)
                if e.sp.result() is None:
                    round_bad += not same_bits(v, ref)
                g = Shewchuk()
                g.p = list(e.p)
                cost = 0
                while g.p:
                    r, st = fsum_round(g.p)
                    cost += st
                    if r == 0.0:
                        break
                    g.add(-r)
                    g.p = [q for q in g.p if q != 0.0]
                canon["shewchuk"].add(cost + g.two_sums)
    # a long stream, so the superaccumulator's flushes show in the amortized count
    n_long = 50_000 if quick else 200_000
    # exponents -4..4, both signs: few entries, so each fills and flushes often
    long_xs = [float(v) for v in np.ldexp(rng.uniform(-2, 2, n_long), rng.integers(-4, 5, n_long))]
    s = SuperAcc()
    for x in long_xs:
        s.add(x)
    long_sup = s.chunk_ops / n_long
    k_long = run_acc(Kulisch, long_xs)
    long_kul = 2 + k_long.ripple / n_long
    # merges: shard states of 1000 values, merged by a random tree
    mrg = {"shewchuk": Stat(), "superacc": Stat()}
    for t in range(4 if quick else 12):
        xs = [gen_mixed, gen_cancel, gen_range][t % 3](rng, 1000)
        split = make_split(len(xs), prng)
        sh = [run_acc(Shewchuk, [xs[i] for i in p]) for p in split]

        def m_sh(a, b):
            t0 = a.two_sums
            a.merge(b)
            mrg["shewchuk"].add(a.two_sums - t0)
            return a

        random_tree(sh, m_sh, prng)
        sa = [run_acc(SuperAcc, [xs[i] for i in p]) for p in split]

        def m_sa(a, b):
            c0 = a.chunk_ops
            a.merge(b)
            mrg["superacc"].add(a.chunk_ops - c0)
            return a

        random_tree(sa, m_sa, prng)
    check(
        "limb-wise rounding (the counted algorithm)",
        "round_to_f64 and float(Fraction(exact sum))",
        f"{round_bad} mismatches over {n_sums} sums x 3",
        "0",
        round_bad == 0,
    )
    print(f"  data: {len(GENS)} generators x {trials} sums of 1 to 1000 values ({n_sums} sums);")
    print("  merges: shards (2..16) of 1000-value sets merged by a random tree")
    print()
    print("  plain float : add 1 fp op; decayed add +1 fp multiply; merge 1 fp op;")
    print("                round 0; bytes 8, fixed")
    print(
        f"  Shewchuk    : add = two-sums (one per component, {SHEWCHUK_FP_PER_TWO_SUM} fp ops"
        f" each): {add['shewchuk']}"
    )
    print(f"                components after a sum: {comps}")
    print(f"                merge two-sums: {mrg['shewchuk']}")
    print(f"                round (fsum's last step) two-sums: {rnd['shewchuk']}")
    print(f"                to canonical form (for a saved state) two-sums: {canon['shewchuk']}")
    print("                bytes 8 x components + 2; canonical <= 40 components = 322 B;")
    print("                raw: bounded by the count of adds (and the 2,098-bit range)")
    print(
        f"  Kulisch     : add {KULISCH_ADD_WORD} word ops + limb ops (2 + ripple):"
        f" {add['kulisch']} (worst 34); {long_kul:.4f} on a stream of {n_long:,}"
    )
    print(f"                merge {K_LIMBS} limb ops + 1 flag OR, fixed")
    print(
        f"                round limb ops: {rnd['kulisch']}"
        f" (worst 1 sign + 34 negate + 34 scan + 2 window + 32 sticky = 103)"
    )
    print("                to canonical form: 0 (the limbs are canonical)")
    print(f"                bytes 8 x {K_LIMBS} + 1 = {Kulisch.state_bytes()}, fixed")
    print(
        f"  superacc    : add {SUPERACC_ADD_WORD} word ops; chunk ops per add (flushes,"
        f" carries, amortized): {add['superacc']} on sums of <= 1000;"
        f" {long_sup:.4f} on a stream of {n_long:,} over 9 exponents"
    )
    print(
        f"                flush ({SUPERACC_FLUSH_WORD} word ops + <= 3 chunk ops) once per"
        " 1024 adds to an entry; live entries after a sum: "
        f"{entries_live}"
    )
    print(f"                merge chunk ops (flush, carry both, add, carry): {mrg['superacc']}")
    print(
        f"                to canonical form chunk ops: {canon['superacc']};"
        f" round (canonicalise + digits): {rnd['superacc']}"
    )
    print(f"                canonical chunks after a sum: {chunks_live}")
    print("                bytes: canonical 9 x chunks + 5 <= 69 x 9 + 5 = 626;")
    print("                live 12 x entries + 9 x chunks + 5 <= 2048 x 12 + 69 x 9 + 5 = 25,202")
    print()
    print("  decayed add, per field: +1 fp multiply fl(p * g) before the add, for every")
    print("  accumulator (the rows' common-clock form, the one that merges exactly); the")
    print("  power p = lam^(T - t) is one libm call per row, shared by all fields.")
    print("  Decaying the state instead (S <- lam^dt S + g) is not exact: per row, plain float")
    print("  1 multiply; Shewchuk 1 multiply per component; Kulisch 34 limb x 53-bit")
    print("  multiplies and a rounding (a 34-limb shift when lam^dt = 2^-n);")
    print("  superacc one multiply per live entry and chunk.")
    print()
    print("  per Gram of k features (1 + k + k(k+1)/2 summed fields: W, S1, S2):")
    for k in (2, 5, 20):
        nf = 1 + k + k * (k + 1) // 2
        print(
            f"    k = {k:>2}: {nf:3d} fields -> float {8 * nf:6,d} B, {nf:4d} merge ops;"
            f" Kulisch {Kulisch.state_bytes() * nf:7,d} B, {K_LIMBS * nf:5,d} merge limb ops;"
            f" superacc canonical <= {626 * nf:7,d} B"
        )


def report_checks() -> int:
    print()
    print("######## proposed tolerances, asserted ########")
    failed = 0
    for claim, oracle, measured, tol, ok in CHECKS:
        failed += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {claim}")
        print(f"         oracle: {oracle}; measured: {measured}; tolerance: {tol}")
    print(f"  {len(CHECKS) - failed} passed, {failed} failed")
    return failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "command", nargs="?", default="all", choices=["accumulators", "merge", "costs", "all"]
    )
    ap.add_argument("--quick", action="store_true", help="fewer trials (about 5 s)")
    args = ap.parse_args()
    t0 = time.time()
    if args.command in ("accumulators", "all"):
        run_accumulators(args.quick)
    if args.command in ("merge", "all"):
        if args.command == "all":
            print()
        run_merge(args.quick)
    if args.command in ("costs", "all"):
        if args.command == "all":
            print()
        run_costs(args.quick)
    failed = report_checks()
    print(
        f"\n(total {time.time() - t0:.1f} s, seed {SEED},"
        f" Python {sys.version.split()[0]}, numpy {np.__version__})"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
