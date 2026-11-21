"""Task 528, part 1: exact accumulators for sums of f64, and shard invariance.

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
Run: ``.venv/bin/python scripts/study_528.py`` (about 20 s).
"""

from __future__ import annotations

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

    def add(self, x: float) -> None:
        if self.sp.take(x):
            return
        p = self.p
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
        # CPython fsum's final step: add from the top until a nonzero error,
        # then correct a half-way case to round-half-even using the sign of
        # the next lower component.
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
                if lo != 0.0:
                    break
            if n > 0 and ((lo < 0.0 and p[n - 1] < 0.0) or (lo > 0.0 and p[n - 1] > 0.0)):
                y = lo * 2.0
                x = hi + y
                yr = x - hi
                if y == yr:
                    hi = x
        return hi + 0.0  # -0.0 -> +0.0 as fsum

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


def random_merge(states: list, rng: random.Random):
    """Merge in a random order and a random grouping (a random binary tree)."""
    pool = list(states)
    rng.shuffle(pool)
    while len(pool) > 1:
        i, j = rng.sample(range(len(pool)), 2)
        a, b = pool[i], pool[j]
        a.merge(b)
        pool = [s for t, s in enumerate(pool) if t not in (i, j)] + [a]
    return pool[0]


def float_random_merge(parts: list[float], rng: random.Random) -> float:
    pool = list(parts)
    rng.shuffle(pool)
    while len(pool) > 1:
        i, j = rng.sample(range(len(pool)), 2)
        v = pool[i] + pool[j]
        pool = [s for t, s in enumerate(pool) if t not in (i, j)] + [v]
    return pool[0]


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
    total = 0
    float_split_diff = 0
    float_set_diff = 0
    for t in range(n_sets):
        xs = gens[t % 3](nrng, 1000)[:1000]
        base = {k: run_acc(cls, xs) for k, cls in ACCS.items()}
        base_r = {k: f64_bits(a.round_to_f64()) for k, a in base.items()}
        base_c = {k: a.canonical() for k, a in base.items()}
        base_raw = {"shewchuk": tuple(base["shewchuk"].p)}
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
                if f64_bits(m.round_to_f64()) != base_r[k]:
                    diff_result[k] += 1
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
            f"  {k:<9} rounded result differs: {diff_result[k]}/{total};"
            f" canonical state differs: {diff_state[k]}/{total}{extra}"
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
    for gname in ("mixed", "cancelling", "huge range"):
        xs = GENS[gname](nrng, 1000)
        a = run_acc(Shewchuk, xs)
        b = run_acc(SuperAcc, xs)
        bc = b.copy()
        bc._flush_all()
        bc._carry()
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


def report_speed(seed: int) -> None:
    print()
    print("== Per-add time in CPython (relative only; says nothing about a Rust cost) ==")
    xs = gen_mixed(np.random.default_rng(seed), 20000)
    t0 = time.perf_counter()
    s = 0.0
    for x in xs:
        s += x
    base = time.perf_counter() - t0
    print(f"  plain float      {1e6 * base / len(xs):7.3f} us/add")
    for k, cls in ACCS.items():
        t0 = time.perf_counter()
        run_acc(cls, xs)
        dt = time.perf_counter() - t0
        print(f"  {k:<16} {1e6 * dt / len(xs):7.3f} us/add  ({dt / base:6.1f}x)")


def main() -> None:
    quick = "--quick" in sys.argv
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    check_correctness(40 if quick else 200, rng)
    check_shards(6 if quick else 24, 5 if quick else 12, SEED)
    report_state(SEED)
    report_speed(SEED)
    print(
        f"\n(total {time.time() - t0:.1f} s, seed {SEED},"
        f" Python {sys.version.split()[0]}, numpy {np.__version__})"
    )


if __name__ == "__main__":
    main()
