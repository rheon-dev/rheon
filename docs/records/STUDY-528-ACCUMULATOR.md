# Study 528, part 1: an exact accumulator for shard invariance

Status as of 2026-11-21: **part 1 of 3 done** (the three accumulators, their
correctness against `math.fsum` and `Fraction`, shard invariance, state bytes).
Parts 2 and 3 are not in this record. Python only: no Rust was built.
Script: `scripts/study_528.py` (seed 528, about 9 s; `--quick` about 1.5 s).
Every input is generated from the seed; nothing is written to disk.

**The question.** Task 528 merges the saved states of shards into the state of
their union and wants the merged state bit-identical whatever the split and
the merge order. Floating-point addition is not associative, so the sums in
the state need an accumulator whose result depends on the exact value of the
sum alone. Which one?

**The design note** for 528 (`vm-tools/research/528-SHARD-DESIGN.md` on
`origin/vm-tools/oracle-gap-scan`) **does not exist**; that branch mentions 528
only in passing (a HyperLogLog row in `494-MODELS-SCAN.md`). This study works
from the task text in `docs/PLAN.md`: exact accumulation is needed for
merge-order invariance; the scan's estimate is 4 to 15 times the per-row cost
and 9 to 12 times the state.

## Answer

All three are exact, round correctly and are shard-invariant in their rounded
result. They differ in what the *state* is:

| | Kulisch (34 x 64-bit limbs) | Sparse superaccumulator | Shewchuk expansion |
|---|---|---|---|
| mismatches vs `fsum` / vs `Fraction`, 1,200 sums | 0 / 0 | 0 / 0 | 0 / 0 (raises on 189 near-overflow sums, as `fsum` does) |
| rounded result differs across 288 sharded merges | 0 | 0 | 0 |
| state bit-identical across those merges | yes, always (the limbs are canonical) | yes, after flush and carry | **no** (raw partials differed in 288 of 288); yes only after canonicalising, which is a loop of exact roundings |
| state bytes | **273, fixed** | 59 to 608 canonical on these data, **626 worst**; live (unflushed) up to 25,202 | 66 to 522 raw on these data, raw unbounded by 40 (77 seen); canonical **314 worst** |
| near-overflow (exact sum finite, a partial sum is not) | exact | exact | raises `OverflowError` |
| CPython per add (relative only) | 63x a float add | 36x | 65x |

**Recommendation for part 2: the Kulisch accumulator.** Its state is one fixed
layout of 273 bytes, canonical by construction, so a merged state re-saves
byte for byte whatever the merge (rule 5's fixtures need exactly that), and a
merge is 34 limb additions. The superaccumulator is smaller on narrow-range
data and probably faster per add (no carry ripple in the common path), but
its saved form must be canonicalised first and its live form is variable and
can be large. Shewchuk's state is not canonical without an extra loop, and it
cannot hold an intermediate above the f64 range.

The per-add timings are CPython's and say nothing about Rust; part 2 or 3
measures the real per-row cost against the scan's 4 to 15 times.

## The math

Every finite f64 is an integer multiple of 2^-1074 (the smallest subnormal),
and every |x| < 2^1024. So x = N * 2^-1074 with |N| < 2^2098, and a sum of n
f64s is an integer below n * 2^2098 in the same units. That single fact is
what the two integer accumulators use; the expansion uses floats.

### (a) Shewchuk's expansion

Shewchuk, J. R. (1997), "Adaptive precision floating-point arithmetic and fast
robust geometric predicates", *Discrete & Computational Geometry* 18(3),
305-363 (cited from memory). An expansion is a list of f64s e_1 .. e_m whose
exact sum is the value, *non-overlapping*: the bit ranges of the components
are disjoint, smallest first.

- **Two-sum.** For |a| >= |b|: hi = fl(a + b), lo = b - (hi - a). Then
  a + b = hi + lo exactly (Dekker's fast two-sum; exact in round-to-nearest,
  subnormals included, unless hi overflows).
- **Grow-expansion** (add x): two-sum x through every component in turn,
  keep each nonzero lo in place, carry hi on, and append the last hi on top.
  Zero elimination keeps the list short. This is exactly CPython's
  `math.fsum` partials loop (`msum`).
- **Merge**: grow the expansion by each component of the other. The value is
  exact, but the list depends on the order: in this study the raw partials
  differed in all 288 sharded merges, though the value never did.
- **Compress** (Shewchuk's routine) renormalises an expansion so each
  component is within an ulp of the next. Taken to its limit it gives the
  greedy form c_1 = round(S), c_2 = round(S - c_1), ..., which depends on S
  alone; `canonical()` computes it, each residue formed exactly by growing
  the expansion by -c. Each c_{i+1} is at most half an ulp of c_i, so there
  are at most 40 components (2098 / 53 + 1); 39 were reached.
- **Length.** The raw list's components need not be 53 bits apart, so its
  length is bounded only by the bit range (about 2,100 one-bit components)
  and the number of adds, not by 40: 77 were seen on 1,000 full-range values.
- **Overflow.** A two-sum whose hi overflows has no exact lo. The list cannot
  hold M + M - M (M the largest f64) though the sum is M; like `fsum`, the
  implementation raises `OverflowError` (189 of 200 near-overflow sums).

### (b) Kulisch long accumulator

Kulisch's exact scalar product (Kulisch and Miranker, *Computer Arithmetic in
Theory and Practice*, 1981; Kulisch, *Computer Arithmetic and Validity*, 2008;
cited from memory): a fixed-point register wide enough for every product of
two floating-point numbers, with headroom for carries.

- **Width.** For sums of f64: 2098 bits of range + 64 bits of count headroom
  (2^64 adds of the largest f64 cannot reach the sign) + 1 sign bit = 2163
  bits, rounded up to **34 limbs of 64 bits = 2176 bits, 272 bytes**, plus one
  flag byte for the non-finite flags: **273 bytes**. N is held in two's
  complement mod 2^2176.
- **For exact dot products** (a Gram entry is a sum of x_i x_j), a product of
  two f64s can be as small as 2^-2148 and as large as 2^2048: 4196 + 64 + 1
  bits = **67 limbs = 536 bytes** (Kulisch's classic register is about 4,288
  bits for IEEE double, from memory). Alternatively each product is split
  exactly into two f64s by FMA (two-product) and both go into the 34-limb
  register; that is exact except where the low half underflows (products
  below about 2^-969), so the 67-limb form is the one that is exact
  everywhere.
- **Add x.** Decompose |x| = m * 2^(p - 1074) with m < 2^53 and p >= 0 (a
  normal number: m includes the hidden bit, p = exponent field - 1; a
  subnormal: m is the fraction, p = 0, so subnormals need no special case).
  Then |x| * 2^1074 = m << p. Limb i = p div 64 takes the low 64 bits of
  m << (p mod 64), limb i + 1 the rest (m << s < 2^117: two limbs). A
  positive x adds with carry, a negative one subtracts with borrow.
- **Carry handling.** Immediate ripple: the carry or borrow out of limb i + 1
  moves up limb by limb until it is absorbed, wrapping mod 2^2176. A long
  ripple needs a run of all-ones (or all-zeros on a borrow) limbs, so it is
  rare except when the sum changes sign (then it runs to the top). A delayed
  carry (each limb holding fewer payload bits) would remove the ripple at
  the cost of a normalisation before save; the superaccumulator below does
  that.
- **Merge.** Add the other's 34 limbs with carry, OR the flags.
- **Canonical.** The limbs are the value: one value, one bit pattern.

### (c) Two-level sparse superaccumulator

Neal, R. M. (2015), "Fast exact summation using small and large
superaccumulators", arXiv:1505.05571 (cited from memory). Neal's small
superaccumulator is 67 signed 64-bit chunks, each nominally holding 32 bits
of the number, so carries can be deferred; his large superaccumulator is 4096
entries indexed by the sign and exponent bits, each summing mantissas of one
exponent and flushed into the small one when its count would overflow. The
version here makes both levels sparse maps:

- **Large level**: exponent field (0..2047) -> [signed sum of 53-bit
  mantissas, count]. 1024 mantissas stay below 2^63, so an entry flushes when
  its count reaches `large_limit` <= 1024 (the study varies it: 1, 3, 7, 1024).
- **Small level**: chunk index c -> signed int64 holding bits 32c .. 32c + 31
  of N. A flushed entry (|s| < 2^63 at position p) adds signed 32-bit pieces
  of |s| << (p mod 32) (< 2^94) into at most three chunks.
- **Carry interval.** A piece is below 2^32, so a chunk starting in
  [-1, 2^32) takes 2^31 - 1 pieces before it could leave int64. Carries
  propagate every `carry_every` <= 2^31 - 1 flushes (the study varies it: 1,
  5, 2^31 - 1): ripple from the lowest chunk, each keeping its low 32 bits
  and passing the arithmetic shift up; past the top the carry settles to 0 or
  -1 (the sign extension), and a -1 becomes the top chunk, moved down over any
  all-ones chunks so the digits are minimal.
- **Merge.** Flush and carry both, add chunk by chunk, carry again; OR the
  flags.
- **Canonical.** Flush every large entry and carry: digits in [0, 2^32), plus
  a top -1 for a negative sum. This is a function of N alone, so the saved
  form is bit-identical across merges (0 of 288 differed, with flush and carry
  intervals varied per shard). The live form is not canonical.
- **Chunks.** |N| < 2^2162 needs chunks 0..67, plus a top -1: at most 69.

### Rounding to f64

**Kulisch and the superaccumulator** round N * 2^-1074 directly: let L be the
bit length of |N|. If L <= 53, the value is exactly representable (normal or
subnormal) and is `ldexp(|N|, -1074)`. Otherwise shift by L - 53, take the
53-bit quotient q and the remainder r: round up if r > half, or r = half and q
is odd (**round half to even**); a carry to 2^53 renormalises. L > 53 means
the value is at least 2^-1021, normal, so 53 bits is the right precision. A
result at or above 2^1024 - 2^970 (the largest f64 plus half its ulp, a tie
that rounds to the even 2^1024) is +-inf. This is correct rounding of the
exact sum.

**Shewchuk** uses `fsum`'s final step: add components from the top down until
a nonzero rounding error appears; if the error is exactly half an ulp and the
next lower component has the same sign, the true sum is past the half-way
point, so round away (the `lo * 2` test). This is correct rounding with ties
to even, as `fsum` documents.

**Against `math.fsum`.** `fsum` is the Shewchuk expansion plus that rounding,
so all three agree with it bit for bit wherever it returns. They differ from
it in three places, by choice:

- `fsum` raises `OverflowError` when a partial sum overflows though the exact
  sum is finite; Kulisch and the superaccumulator return the correctly
  rounded value (the Shewchuk implementation raises, as `fsum`).
- Non-finite inputs: all three keep three flags (NaN, +inf, -inf) that merge
  by OR; the result is NaN if NaN or both infinities were added, else the
  infinity. `fsum` raises `ValueError` on inf + -inf; IEEE and this choice
  give NaN.
- Zero: an exact zero sum is +0.0, as `fsum` returns (`fsum([-0.0])` is
  `0.0`); IEEE's sequential -0.0 + -0.0 is -0.0. Keeping that would need one
  more flag ("every input was -0.0"), which also merges by AND.

## Results (seed 528)

### Correctness: rounded result against `fsum` and `float(Fraction(exact sum))`

200 sums of 1 to 1,000 values per generator; cells are mismatches against
`fsum` / against `Fraction`; `float(Fraction)` is correctly rounded in
CPython, and an `OverflowError` from it is read as the infinity of the sum's
sign. Bits compared with `struct.pack('<d')`.

| generator | Shewchuk | Kulisch | superacc |
|---|---|---|---|
| mixed (exponents -60..60, both signs) | 0 / 0 | 0 / 0 | 0 / 0 |
| cancelling (v and -v pairs, large terms 2^60..2^200, small 2^-200..2^-60, a few nudged an ulp) | 0 / 0 | 0 / 0 | 0 / 0 |
| huge range (random bit patterns, subnormals to 2^1012) | 0 / 0 | 0 / 0 | 0 / 0 |
| subnormal (integers times 2^-1074, plus the smallest normal) | 0 / 0 | 0 / 0 | 0 / 0 |
| ties (a + half an ulp of a, +- a hair or exactly, among cancelling noise) | 0 / 0 | 0 / 0 | 0 / 0 |
| near overflow (M + M - M + noise, sometimes + half an ulp of M) | 0 / 0, raised 189 | 0 / 0 | 0 / 0 |

`fsum` raised on the same 189 near-overflow sums (compared against `Fraction`
only there). Ten special cases (infinities, NaN, signed zeros, the empty sum,
subnormal cancellation) gave 0 mismatches against the policy above.

### Shard invariance

24 sets of 1,000 values (mixed, cancelling, huge range in turn), each split 12
times into 2 to 16 shards of random sizes from a shuffled order (shards in no
time order), each shard accumulated, then merged in a random order and a
random grouping (a random binary tree): 288 sharded merges per accumulator.

- Rounded result differing from the unsharded run: **0 / 288 for all three.**
- Canonical state differing: 0 / 288 for all three. Raw Shewchuk partials
  differed in 288 / 288.
- **Plain float sum** (each shard summed in order, shard sums merged by the
  same random tree), against the sequential sum: **differs in 263 / 288
  splits; 24 / 24 sets** had at least one differing split.

### State bytes

| | formula | these data (1,000 values) | worst case |
|---|---|---|---|
| Kulisch | 34 x 8 + 1 | 273 B | 273 B, fixed (536 B for exact products) |
| Shewchuk raw | 8 x components + 2 | 66 (mixed), 114 (cancelling), 522 (huge range); 618 at most over 20 full-range sets | bounded by the bit range and the count of adds, not by 40 |
| Shewchuk canonical | same | at most 314 over 20 full-range sets | 40 components, 322 B (39 reached: 314 B) |
| superacc live | 12 x large entries + 9 x chunks + 5 | 1,457 (mixed), 1,601 (cancelling), 9,269 (huge range) | 2048 x 12 + 69 x 9 + 5 = 25,202 B |
| superacc canonical | 9 x chunks + 5 | 59, 86, 608 | 69 x 9 + 5 = 626 B |

Each figure is per accumulated scalar. A state with k coefficients' worth of
sums (a Gram is k(k+1)/2 entries) multiplies it: Kulisch's 273 B against an
f64's 8 B is 34 times the state per entry (536 B, 67 times, for exact
products), well above the scan's 9 to 12 times. The fixed 273 B is the price
of a canonical layout; the superaccumulator's canonical form is smaller only
when the data's exponents span little of the range.

### CPython time per add (relative only)

Plain float 0.015 us; Shewchuk 0.99 us (65x); Kulisch 0.97 us (63x);
superacc 0.55 us (36x). Interpreter overhead dominates all three; these
numbers say nothing about Rust.

## What is missing (parts 2 and 3)

The per-row cost in a compiled implementation, whether the accumulator sits
on the per-row path or only in the merge, and how it fits rule 5's fixtures
and the state-size rule (a growth of this size is the user's call before it
lands).
