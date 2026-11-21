# Study 528: an exact accumulator for shard invariance

Status as of 2026-11-21: **parts 1, 2 and 3 of 3 done; the study is complete**
(part 1: the three accumulators, their correctness against `math.fsum` and
`Fraction`, shard invariance, state bytes; part 2: which state fields need
one, and decay in shard merges; part 3: one script, the operation and byte
table, the oracle for each claim, the proposed tolerances and the
recommendation). Python only: no Rust was built. One open question is the
user's (part 3, section 5). Script: `scripts/study_528.py`, subcommands
`accumulators` (part 1), `merge` (part 2), `costs` (part 3) and `all` (the
default), with `--quick` (seed 528; `all` about 20 s, `--quick` about 5 s;
it exits 1 if a proposed tolerance fails). Every input is generated from the
seed; nothing is written to disk.

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

The per-add timings are CPython's and say nothing about Rust; part 3 counts
operations instead (its table), and no compiled timing was made.

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

(Part 1's run. The consolidated script no longer times anything: part 3
counts operations instead.) Plain float 0.015 us; Shewchuk 0.99 us (65x); Kulisch 0.97 us (63x);
superacc 0.55 us (36x). Interpreter overhead dominates all three; these
numbers say nothing about Rust.

# Part 2: which states need it, and decay in shard merges

Script: `scripts/study_528.py merge` (written as `scripts/study_528_merge.py`
and folded into the one script in part 3, numbers unchanged; it uses part 1's
`Kulisch` unchanged; seed 528, about 8 s; `--quick` about 2 s). State fields read from
`origin/main` (`ewridge.rs`, `gaps.rs` `Acc`/`Cross`, `ewcov.rs` `EwCov`,
`rls.rs` `Rls`): fields and update equations only.

## Answer

1. **None of the three families keeps raw sums today.** `ew_cov` and
   `ewridge`'s Grams are mean form (weighted Welford: `a = lam W/(lam W + w)`,
   `b = w/(lam W + w)`, `C <- a C + a b d d'`, `m <- m + b d`), and `rls`
   keeps a Cholesky factor `R` of `A` and `u = R^-T b`. A mean-form merge is
   Chan's formula (divisions), an `R` merge is a QR of `[R_A; R_B]` (Givens
   rotations): neither is exact with any accumulator. To merge exactly, the
   state must hold the **raw weighted sums** and the accumulator holds those.
2. **Without decay** raw sums merge exactly with Kulisch: 0 of 192 splits
   differ (k = 2 and 5), against every split differing with plain floats.
3. **With decay** the merge `S_A lam^(T_B - T_A) + S_B` is exact only when
   the factor is an exact power of two (or 1: one common clock). A general
   `lam^dT` is a rounded value: up to 13 ulps off with the accumulator.
4. A read of the centred covariance from raw sums is exact only if the
   **products** are exact too (FMA split into the register; 67-limb width
   for full range): 0 ulps at offset 1e8, where per-row rounded products
   lose the covariance entirely (relative error 1.2).

## 1. Which state fields are sums

| family | sums of products (a merge must add these exactly, once held raw) | weight sums (sums of f64: exact only with the accumulator, unless every w is a small integer and nothing decays) | not sums: merged by another rule |
|---|---|---|---|
| `ew_cov` (`EwCov`) | `m` + `m_lo` (EW mean, raw form `S1 = Σ w x`); `c` (centred co-moments, k(k+1)/2, raw form `S2 = Σ w x x'`) | `w_sum` (`W = Σ lam^.. w`), `q_sum` (`Σ lam^2.. w^2`) | `prior_scale` (product of decay factors = lam^elapsed: follows the merged clock); `rows_learned` (u64 count: integer add, exact); `runs` (per-feature held value since a start row: kept if equal in both, else restarted); `precision_prior`, `precision_scale` (config, and the prior's scale decays with `a`, a ratio, not a sum); `pending` (flush before merge) |
| `ewridge` (`Acc` per Gram, plus `slow` twin) | each Gram as `ew_cov`; `cross.m`, `cross.mj` (per-target feature means), `cross.my` (target means), `cross.c` (centred `Σ w (x - m_j)(y - ybar_j)`, k per target), `tm` var (`Σ w y^2` raw) | `wj` (per target), `cross.w` (= `n_eff`), `tm.q` (`Σ w^2`), `weight_since_solve` | `cross.n`, `cross.nj`, `rows_since_solve`, `solve_failures` (u64: integer add); `since_solve` and `win.clock` (the decayed clock, a `Stamp`: the later of the two, not a sum); `wsig`/`sig2` (residual variance: each residual uses the shard's own fit at that row, so it is **path dependent and not mergeable** at all; a merged state needs a different definition, e.g. recomputed from the merged Gram); `beta`, `ready`, factors (derived: re-solve after the merge); `win.snaps` (a window's ring of snapshots: each would need the same treatment, and a merge must line up their clocks) |
| `rls` (`Rls`) | none held as sums: `A = delta lam^n I + Σ lam w z z'` and `b_j = Σ lam w y_j z` live only inside `r` (`A = R'R`) and `u_j = R^-T b_j` | `w_sum`, `w_target`, `s2` (`Σ lam^2 w^2`) | `r`, `u` (merge by re-triangularizing the stack: rotations, rounded); `beta` (derived); `seen` (OR). As raw sums, the prior `delta I` is in every shard's `A`: a merge must count it once (subtract the extra `delta lam^.. I`s), and it decays with the clock |

So with raw sums, what a shard merge must add exactly per Gram over k
features (targets are k extra columns of the same Gram): `W`, `Q`, the k
first moments `S1`, the k(k+1)/2 second moments `S2`, and per target `W_j`,
`Σ w y_j`, `Σ w y_j x` (k), `Σ w y_j^2`. Counts add as integers (already
exact); clocks take the later value; derived fields are recomputed.

## 2. Merge without decay

Each row's terms are rounded once per row (`w`, `fl(w x_i)`,
`fl(fl(w x_i) x_j)`), so they do not depend on the split. 8 sets of 500 rows
(weights in [0.5, 2), features at levels 0, 3, 1e3, scales 1e-2..1e2), each
split 12 ways into 2 to 16 shards of random size from shuffled rows, shards
merged by a random binary tree; compared with the unsharded state:

| k | fields | Kulisch sums (limbs) | plain float sums | mean form, Chan merge |
|---|---|---|---|---|
| 2 | 6 | splits 0 / 96, fields 0 / 576 | splits 96 / 96, fields 512 / 576, worst 18 ulps | 96 / 96, 538 / 576 |
| 5 | 21 | splits 0 / 96, fields 0 / 2016 | splits 96 / 96, fields 1846 / 2016, worst 232 ulps | 96 / 96, 1927 / 2016 |

(The mean form's worst ulps, up to 1.1e7, are on co-moments near zero; it is
there to show rheon's current layout is not merge-invariant either.)

**Raw sums and centring.** The mean form exists because raw float sums lose
a covariance under an offset (offset 1e8, k = 2): plain float raw sums gave
6 for a true -0.01306 (relative error 460); the weighted Welford gave a
relative error 1.6e-7. Exact sums of the per-row *rounded* products are
invariant but still wrong (relative error 1.2: each product's rounding at
1e16 is 1 unit). Exact sums of **exact products** (each `w x_i x_j` split by
FMA into four f64s, all added to the register) and one rounding at the read
(`S2/W - S1 S1'/W^2` in rationals) gave the true value to **0 ulps**. So the
accumulator makes raw sums better than the mean form, but only with exact
products: four register adds per entry per row, and the 67-limb (536 B)
register where products can leave the 34-limb range (below about 2^-969).

## 3-4. Merge with decay

A decayed sum at clock T is `S(T) = Σ lam^(T - t) w_t g_t`. Two shards at
clocks T_A <= T_B merge as `S_A(T_A) lam^(T_B - T_A) + S_B(T_B)`, which is the
sum at T_B. Reference: the exact (Kulisch) sum of every row's
`fl(lam^(T_end - t) g_t)`, one common clock. `lam = 2^(-1/h)`, h = 50 ticks,
integer clock with gaps of 1 to 3 ticks, each power one libm `pow` from the
exact integer offset. 4 sets x 8 splits per case, k = 2 and 5 (192 and 672
fields):

| case | k = 2: splits, fields, worst | k = 5 |
|---|---|---|
| common clock T_end, interleaved shards, add only | **0 / 32**, 0 / 192, 0 | **0 / 32**, 0 / 672, 0 |
| contiguous, own clocks, register rounded to f64 then x `lam^dT` | 24 / 32, 43, 3 ulps | 29 / 32, 171, 9 ulps |
| contiguous, own clocks, register x `lam^dT` rounded once to the grid | 22 / 32, 41, 5 ulps | 27 / 32, 154, 7 ulps |
| contiguous, own clocks, plain floats | 32 / 32, 117, 13 ulps | 32 / 32, 436, 17 ulps |
| contiguous, every row time a multiple of h (every power is 2^-n) | **0 / 32** | **0 / 32** |
| contiguous, shard clocks on the h grid (`lam^dT` = 2^-n), rows free, plain `pow` | 5 / 32, 5, 1 ulp | 15 / 32, 49, 3 ulps |
| interleaved, own clocks, register x `lam^dT` rounded once | 29 / 32, 55, 5 ulps | 30 / 32, 183, 13 ulps |
| interleaved, shard clocks on the h grid, plain `pow` | 1 / 32, 1, 1 ulp | 12 / 32, 30, 2 ulps |
| interleaved, own clocks, plain floats | 32 / 32, 101, 11 ulps | 32 / 32, 372, 46 ulps |
| contiguous, shard clocks on the h grid, power split by whole half-lives | **0 / 32** | **0 / 32** |
| interleaved, shard clocks on the h grid, power split by whole half-lives | **0 / 32** | **0 / 32** |
| (unsharded streaming recursion `S <- lam^dt S + g`, floats, against the reference) | 4 / 4 sets, 21 / 24, 3 ulps | 4 / 4, 69 / 84, 16 ulps |

**Why.** The power is rounded: `fl(fl(lam^a) fl(lam^b)) != fl(lam^(a+b))` in
15,545 of 20,000 random pairs, and even the exact product of the two rounded
powers differs from `fl(lam^(a+b))` in 19,593. So rescaling a shard's state
by `lam^(T_B - T_A)` cannot reproduce the rows' own powers from the common
clock, however exactly the register is multiplied (multiplying the 2176-bit
register by a 53-bit factor exactly and rounding once to the 2^-1074 grid
still left 154 of 672 fields off).

**What is exact:**

- **Rows in one common clock**: every shard adds `fl(lam^(T_ref - t) g_t)`
  for one `T_ref` all shards know (the merged state's clock, an epoch), any
  sharding, interleaved or contiguous: a merge is an add. The cost is range:
  `lam^(T_ref - t)` underflows past 1074 half-lives back and overflows past
  1024 ahead, so `T_ref` must stay within that of every row (re-anchoring is
  a rescale, and so rounded).
- **Clock offsets of whole half-lives**: `lam^(nh) = 2^-n`, a shift of the
  register, exact while no bit falls below 2^-1074. With rows at any time
  this holds to the bit only if each row's power is computed as
  `2^-(d div h) * fl(2^(-(d mod h)/h))` (a whole-half-life split); a plain
  `pow(2, -d/h)` rounds `-d/h` and `-(d + nh)/h` differently and left up to
  15 of 32 splits off by 1 to 3 ulps. With that split, shard clocks on the
  h grid merged exactly, contiguous and interleaved.
- **Contiguous shards with the clock carried in sequence** (shard B starts
  from A's final state) are not a merge: that is the sequential run, the same
  to the bit as the unsharded one, with or without the accumulator.

**What is not exact:** any merge across a clock offset that is not a whole
number of half-lives, contiguous or interleaved (22 to 30 of 32 splits off,
up to 13 ulps with the accumulator; plain floats 32 of 32, up to 46 ulps).
Contiguous and interleaved shards behave alike: the merge's error comes from
the offset's power, not from where the rows sit. The plain float streaming
recursion itself is not the common-clock sum (up to 16 ulps): an exact,
shard-invariant decayed state means a different per-row update, not only a
different merge.

## 5. Operations and bytes per merge, per scalar field

| | operations | bytes |
|---|---|---|
| plain float, no decay | 1 add | 8 |
| plain float, decay | 1 multiply + 1 add (one `pow` per merge, shared) | 8 |
| Kulisch, no decay | 34 limb adds with carry + 1 flag OR | 273 |
| Kulisch, decay, factor 2^-n | a shift of 34 limbs (34 shifts + ORs) + 34 limb adds; exact | 273 |
| Kulisch, decay, general factor, exact multiply | 34 limb x 53-bit multiplies + a 34-limb shift with one rounding + 34 limb adds; not exact | 273 |
| Kulisch, decay, general factor, via f64 | a 34-limb scan to round + 1 multiply + a two-limb add and its carry; not exact | 273 |

Per Gram (W, S1, S2; Q and per-target fields add the same per field):

| k | fields | plain float | Kulisch |
|---|---|---|---|
| 2 | 6 | 48 B, 6 adds | 1,638 B, 204 limb adds |
| 5 | 21 | 168 B, 21 adds | 5,733 B, 714 limb adds |
| 20 | 231 | 1,848 B, 231 adds | 63,063 B, 7,854 limb adds |

Exact products (needed for an exact centred read, section 2) double the
width to 67 limbs (536 B per field; 123,816 B for k = 20). Both are far above
the state-size rule's "reasonable" for k = 20 and are the user's call.

## What part 2 left open

The per-row cost in a compiled implementation, whether the accumulator sits
on the per-row path or only in the merge, how a raw-sum layout fits rule 5's
fixtures, and the state-size question above. Part 3 answers the cost in
counted operations (not compiled timings) and states the state-size question
for the user; the fixtures question follows whichever layout the user takes.

# Part 3: operations and bytes, oracles, tolerances, recommendation

Script: `scripts/study_528.py costs` (and `all`, which runs the three parts
and then asserts every proposed tolerance below: 19 checks, 0 failed; `all`
19.5 s, `all --quick` 4.6 s, exit 0). Parts 1 and 2 reproduce their numbers
unchanged under the one script; part 3 added a `Fraction` oracle beside each
earlier comparison against an unsharded run.

## Answer

- **The table** (section 2) counts operations per add, decayed add, merge,
  save (canonical form) and `round_to_f64`, and the state bytes, per scalar
  sum. No timing claims: Python times the interpreter, not the arithmetic.
- **Every exact claim has 0 mismatches against its oracle** (section 3):
  `math.fsum` and `float(Fraction(exact sum))` for one sum, 1,200 sums per
  accumulator; the exact common-clock sum in `Fraction` for the decayed
  merges.
- **Tolerances** (section 4): zero, bit for bit, for every exact claim; **13
  ulps** for a decayed merge across general clock offsets (the measured
  worst over 320 splits); none for plain floats; the state bytes as exact
  counts (Kulisch exactly 273 B per sum).
- **Recommendation for a Rust port: the Kulisch accumulator** (section 5),
  if any state adopts raw sums. Whether any should is open and is the
  user's decision under the state rule.

## 1. One script

`scripts/study_528_merge.py` is folded into `scripts/study_528.py` and
deleted. `python scripts/study_528.py [accumulators|merge|costs|all]
[--quick]`; `all` is the default. Each claim registers a check (claim,
oracle, measured, tolerance) and the run ends with the list and exits 1 if
one fails. The CPython per-add timing of part 1 is dropped.

## 2. Operations and bytes per scalar sum

Units: an **fp op** is one f64 add, subtract, multiply or compare; a **word
op** one integer operation on a 64-bit word (shift, mask, add, compare); a
**limb op** one add or subtract with carry, negation step or nonzero test of
a 64-bit limb (Kulisch) or of a 64-bit chunk holding 32 payload bits (the
superaccumulator). Fixed word ops are read off each add path; every variable
count (carry ripple, components, flushes, carry steps, scans) is counted
while running the data: 6 generators x 30 sums of 1 to 1,000 values (180
sums), shard merges of 1,000-value sets (2 to 16 shards, random tree), and a
200,000-value stream for the amortized counts. "m" is the expansion's
component count, "E" the superaccumulator's live large entries, "C" its
chunks.

| | add | decayed add (extra) | merge | to canonical form (save) | round_to_f64 | bytes, fixed | bytes, worst | bytes formula |
|---|---|---|---|---|---|---|---|---|
| plain float | 1 fp op | +1 fp multiply | 1 fp op | 0 | 0 | 8 | 8 | 8 |
| Shewchuk expansion | m two-sums of 7 fp ops (3 add/sub, 3 for the magnitude compare, 1 zero test); measured mean 18.1, max 89 two-sums | +1 fp multiply | the other's components added one by one: mean 1,587, max 16,841 two-sums | greedy re-rounding: mean 340, max 2,006 two-sums | fsum's last step: mean 1.6, max 8 two-sums (at most m) | 2 (length, flags); no fixed part | canonical 322 (40 components); raw: no bound below the bit range (77 components, 618 B, seen) | 8m + 2 |
| Kulisch, 34 limbs | 11 word ops + 2 limb ops + carry ripple: mean 2.24, max 34 limb ops on sums of <= 1,000; 2.009 on the 200,000 stream | +1 fp multiply | 34 limb ops + 1 flag OR, fixed | 0 (the limbs are canonical) | mean 25, max 69 limb ops; worst 103 (1 sign + 34 negate + 34 scan + 2 window + 32 sticky), plus about 10 word ops | **273** | **273** | 8 x 34 + 1 |
| sparse superaccumulator | 10 word ops (a direct-indexed large level) + amortized flush of 10 word ops and <= 3 chunk ops per 1,024 adds to one entry: 0 chunk ops on sums of <= 1,000; 0.0024 on the 200,000 stream | +1 fp multiply | flush and carry both, C chunk adds, carry: mean 301, max 2,101 chunk ops | flush E entries, carry: mean 269, max 2,121 chunk ops | canonicalise + digit rounding: mean 284, max 2,193 chunk ops | 5 (two lengths, flags) | canonical 626 (C = 69); live 25,202 (E = 2,048) | canonical 9C + 5; live 12E + 9C + 5 |

**Decayed add.** In the rows' common-clock form (the form that merges
exactly, part 2) every accumulator adds `fl(p g)` with `p = lam^(T_ref - t)`:
one extra fp multiply per field, and one libm power per row shared by every
field (with the whole-half-life split, a divmod, a power and an `ldexp`).
Decaying the state itself each row (`S <- lam^dt S + g`) is not exact for any
of them: plain float 1 multiply; the expansion 1 multiply per component;
Kulisch 34 limb x 53-bit multiplies and a rounding (a 34-limb shift when
`lam^dt = 2^-n`); the superaccumulator one multiply per live entry and chunk.

**Per Gram of k features** (1 + k + k(k+1)/2 summed fields: W, S1, S2;
targets add fields at the same rate):

| k | fields | plain float | Kulisch | superaccumulator, canonical worst |
|---|---|---|---|---|
| 2 | 6 | 48 B, 6 merge ops | 1,638 B, 204 merge limb ops | 3,756 B |
| 5 | 21 | 168 B, 21 merge ops | 5,733 B, 714 merge limb ops | 13,146 B |
| 20 | 231 | 1,848 B, 231 merge ops | 63,063 B, 7,854 merge limb ops | 144,606 B |

Exact products (part 2, section 2) widen Kulisch to 67 limbs: 536 B per
field (123,816 B for k = 20), and four register adds per entry per row.

**Reading the table.** Kulisch's add costs about 2 limb ops on almost every
row (a ripple past the two written limbs averaged 0.24 limb ops on short
mixed sums and 0.009 on the long stream; its worst, a sign change, runs to
the top), and its merge, save and size are fixed. The superaccumulator's add
is cheapest (no limb op in the common path) but every merge and every save
first flushes and carries, hundreds of chunk ops on these data, and its live
working set is large (a dense large level is 2,048 entries). The expansion's
add grows with its length and its merge with the product of the two lengths.

## 3. Oracles, claim by claim

| claim | oracle | trials | largest gap |
|---|---|---|---|
| one sum, each accumulator, rounded | `math.fsum` where it returns, and `float(Fraction(exact sum))` | 1,200 sums per accumulator (Shewchuk: 1,011 compared; it raises on the 189 near-overflow sums on which `fsum` raises) | 0 mismatches, each |
| special values | the stated flag policy (part 1) | 10 cases x 3 | 0 |
| shard merge, no decay, rounded | `float(Fraction(exact sum of the set))`, and the unsharded run | 288 splits per accumulator | 0, each |
| shard merge, no decay, saved state | the unsharded run's canonical state, byte for byte | 288 per accumulator | 0, each (raw expansion partials: 288 differ) |
| raw Gram sums, no decay, every field | `float(Fraction(exact sum of the per-row terms))`; limbs against the unsharded run | 192 splits; 576 (k = 2) + 2,016 (k = 5) field values | 0 |
| decayed sums, unsharded register | `float(Fraction(exact sum of every row's fl(fl(lam^(T_end - t)) g)))`, one common clock | 1,188 field references | 0 |
| decayed merge, common clock or whole-half-life offsets | the same | 256 splits (4 cases x k = 2, 5 x 32) | 0 |
| decayed merge, general clock offsets, with the register | the same | 320 splits (5 cases x k = 2, 5 x 4 sets x 8 splits), 194 off | 13 ulps (k = 2: 5; k = 5: 13) |
| the counted limb-wise rounding (section 2) | `round_to_f64` and `float(Fraction(exact sum))` | 180 sums x 3 | 0 |
| centred covariance from exact sums of exact products | `Fraction` over the data | 1 set, offset 1e8 | 0 ulps |

The decayed oracle cannot be the decayed sum with exact powers (`lam^d` is
irrational); it is the exact sum of what each row contributes on the common
clock, `fl(fl(lam^(T_end - t)) g)`, which `Fraction` holds exactly.

## 4. Proposed tolerances

- **An exact accumulator: zero.** One sum, a shard merge without decay, and
  a decayed merge on one common clock or across whole-half-life offsets
  (with the split power) are bit-identical to the oracle: a test compares
  `to_bits()` for equality and says the tolerance is zero, not a small ulp
  count. The saved state is byte-identical too, for Kulisch's limbs and the
  superaccumulator's canonical form (never for raw expansion partials).
- **A decayed merge across general clock offsets: 13 ulps**, the measured
  worst (an integer, so no rounding up), over 320 splits (5 accumulator
  cases x k = 2 and 5 x 4 sets x 8 splits). It grows with k (5 at k = 2, 13
  at k = 5) and was measured on this generator only, so it holds for k <= 5
  here and must be re-measured for larger k. It is above the 5-ulp line the
  project uses for output-moving changes, so a test that adopts it is the
  user's call.
- **Plain floats: none.** They are the contrast (every split differs).
- **State size, exact counts per scalar sum:** plain float 8 B; **Kulisch
  exactly 273 B** (an equality test); superaccumulator canonical exactly
  9C + 5 B for its C chunks, at most 626 B (608 B seen), live at most
  12E + 9C + 5 = 25,202 B; expansion canonical exactly 8c + 2 B for its c
  components, at most 322 B (314 B reached); raw expansion: no tolerance (its
  length depends on the adds).

## 5. Recommendation, and the open question

**A Rust port should use the Kulisch accumulator.** Its state is one fixed
layout, 273 B per sum, canonical with no save step, so merged states re-save
byte for byte and one fixture shape covers every case; its merge is 34 limb
ops whatever the data; its add is 11 word ops and about 2 limb ops; its
rounding is at most 103 limb ops, once per read. The superaccumulator adds
more cheaply but pays hundreds of chunk ops at every merge and save, keeps a
large live working set, and its saved size varies (59 to 608 B seen, 626 B
worst, above Kulisch's 273 B). The expansion's cost grows with its length,
its raw state is not canonical, and it cannot hold an intermediate above
the f64 range.

**Open: does any state need it?** None holds raw sums today. The three model
families that keep a Gram or a factor (part 2, section 1) would need the
accumulator only if their states were changed to raw weighted sums. That
change multiplies each summed field from 8 B to 273 B (34 times; 536 B with
exact products), for example 63,063 B against 1,848 B per Gram at k = 20,
bounded by k, not by the stream; and with decay it is exact only on a common
clock or whole-half-life offsets, which changes the per-row update as well as
the merge. Under the state rule a growth of that size is raised with the user
before it lands. This study does not decide it.
