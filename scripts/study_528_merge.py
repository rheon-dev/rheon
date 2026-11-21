"""Task 528, part 2: which model states need an exact accumulator, and decay in shard merges.

Uses part 1's Kulisch accumulator (``scripts/study_528.py``) unchanged.

1. Without decay, merging shard states is adding their sums. The merged
   Gram (weight, first moments, second moments of k features) is compared
   with the unsharded state over random shardings: Kulisch against plain
   float sums, and rheon's own mean form (weighted Welford, merged by Chan's
   formula) for contrast.
2. With decay, a shard's sums are taken at its own clock T. Merging two
   needs S_A * lam^(T_B - T_A) + S_B. The study measures what that rounded
   power costs against the exact sum of every row in one common clock, for
   contiguous shards (general and power-of-two offsets) and interleaved ones.
3. Operation and byte counts per merge, per state field.

Python only; every input is generated from a seed; nothing is written to disk.
Run: ``.venv/bin/python scripts/study_528_merge.py`` (about 15 s; ``--quick``
about 3 s).
"""

from __future__ import annotations

import math
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_528 import K_BITS, K_LIMBS, MASK64, Kulisch, f64_bits  # noqa: E402

SEED = 528
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
    return ["W"] + [f"S1_{i}" for i in range(k)] + [f"S2_{i}{j}" for i in range(k) for j in range(i, k)]


def make_split(n: int, rng: random.Random, contiguous: bool) -> list[list[int]]:
    s = rng.randint(2, 16)
    idx = list(range(n))
    if not contiguous:
        rng.shuffle(idx)
    cuts = sorted(rng.sample(range(1, n), s - 1))
    b = [0] + cuts + [n]
    return [sorted(idx[b[t] : b[t + 1]]) for t in range(s)]


def random_tree(items: list, combine, rng: random.Random):
    pool = list(items)
    rng.shuffle(pool)
    while len(pool) > 1:
        i, j = rng.sample(range(len(pool)), 2)
        v = combine(pool[i], pool[j])
        pool = [p for t, p in enumerate(pool) if t not in (i, j)] + [v]
    return pool[0]


# --------------------------------------------------------------------------
# 1. No decay
# --------------------------------------------------------------------------


def kulisch_state(terms: list[list[float]], rows: list[int]) -> list[Kulisch]:
    accs = [Kulisch() for _ in terms[0]]
    for r in rows:
        for a, t in zip(accs, terms[r]):
            a.add(t)
    return accs


def kulisch_merge(a: list[Kulisch], b: list[Kulisch]) -> list[Kulisch]:
    for x, y in zip(a, b):
        x.merge(y)
    return a


def float_state(terms: list[list[float]], rows: list[int]) -> list[float]:
    s = [0.0] * len(terms[0])
    for r in rows:
        s = [si + ti for si, ti in zip(s, terms[r])]
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
    C = [[(Wa * Ca[i][j] + Wb * Cb[i][j]) / W + d[i] * d[j] * (Wa * Wb / (W * W)) for j in range(k)] for i in range(k)]
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
        worst = {"float": 0, "meanform": 0}
        splits = 0
        for _ in range(n_sets):
            w, x = gen_rows(rng, N_ROWS, k)
            terms = [row_terms(w[r], x[r]) for r in range(N_ROWS)]
            allrows = list(range(N_ROWS))
            ref_k = [a.canonical() for a in kulisch_state(terms, allrows)]
            ref_f = float_state(terms, allrows)
            ref_m = flat_mean_form(welford(w, x, allrows))
            for _ in range(n_splits):
                splits += 1
                parts = make_split(N_ROWS, prng, contiguous=False)
                mk = random_tree([kulisch_state(terms, p) for p in parts], kulisch_merge, prng)
                diff = sum(a.canonical() != b for a, b in zip(mk, ref_k))
                tot["kulisch"][0] += diff > 0
                tot["kulisch"][1] += diff
                mf = random_tree(
                    [float_state(terms, p) for p in parts], lambda a, b: [u + v for u, v in zip(a, b)], prng
                )
                diff = [ulps(a, b) for a, b in zip(mf, ref_f)]
                tot["float"][0] += any(diff)
                tot["float"][1] += sum(d > 0 for d in diff)
                worst["float"] = max(worst["float"], max(diff))
                mm = flat_mean_form(random_tree([welford(w, x, p) for p in parts], chan, prng))
                diff = [ulps(a, b) for a, b in zip(mm, ref_m)]
                tot["meanform"][0] += any(diff)
                tot["meanform"][1] += sum(d > 0 for d in diff)
                worst["meanform"] = max(worst["meanform"], max(diff))
        ent = splits * n_fields
        print(f" k = {k}: {n_fields} fields ({field_names(k)[:4]} ...), {splits} splits, {ent} field values")
        print(
            f"   Kulisch sums       : splits differing {tot['kulisch'][0]:4d} / {splits}; fields differing {tot['kulisch'][1]:5d} / {ent} (limbs compared)"
        )
        print(
            f"   plain float sums   : splits differing {tot['float'][0]:4d} / {splits}; fields differing {tot['float'][1]:5d} / {ent}; worst {worst['float']} ulps"
        )
        print(
            f"   mean form + Chan   : splits differing {tot['meanform'][0]:4d} / {splits}; fields differing {tot['meanform'][1]:5d} / {ent}; worst {worst['meanform']} ulps"
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
        f"   exact sums of exact products (FMA split), 1 rounding: {from_exact:.17g}  ({ulps(from_exact, truth)} ulps)"
    )
    print(
        f"   exact sums of per-row rounded products            : {from_rounded:.17g}  (rel. error {rel(from_rounded):.1e})"
    )
    print(
        f"   plain float raw sums                              : {from_float:.17g}  (rel. error {rel(from_float):.1e})"
    )
    print(f"   rheon's mean form (weighted Welford, floats)      : {C[0][1]:.17g}  (rel. error {rel(C[0][1]):.1e})")


# --------------------------------------------------------------------------
# 2. Decay
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
        for a, t in zip(accs, terms[r]):
            a.add(p * t)
    return accs


def decayed_float(terms, ts, rows, T, h) -> list[float]:
    s = [0.0] * len(terms[0])
    for r in rows:
        p = pow_lam(T - ts[r], h)
        s = [si + p * ti for si, ti in zip(s, terms[r])]
    return s


def streaming_float(terms, ts, h) -> list[float]:
    """rheon's own recursion: S <- lam^(dt) S + term, row by row."""
    s, last = [0.0] * len(terms[0]), ts[0]
    for r in range(len(ts)):
        p = pow_lam(ts[r] - last, h)
        s = [p * si + ti for si, ti in zip(s, terms[r])]
        last = ts[r]
    return s


def check_decay(n_sets: int, n_splits: int) -> None:
    print()
    print("== 3-4. Merge with decay: lam = 2^(-1/h), integer clock, 500 rows, k = 2 and 5, 2..16 shards ==")
    print("   reference: the exact (Kulisch) sum of every row's fl(lam^(T_end - t) * term), one common clock")
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
        f"   rounded power: fl(fl(lam^a) * fl(lam^b)) != fl(lam^(a+b)) in {comp_bad} / {trials} pairs (h = {h});"
        f" the exact product != fl(lam^(a+b)) in {comp_exact_bad} / {trials}"
    )

    cases = [
        ("common clock T_end, interleaved shards (add only)", False, "free", "common"),
        ("contiguous, own clocks, register rounded to f64 then * lam^dT", True, "free", "round"),
        ("contiguous, own clocks, register * lam^dT rounded once to the grid", True, "free", "exact"),
        ("contiguous, own clocks, plain float sums * lam^dT", True, "free", "float"),
        ("contiguous, every row time a multiple of h (all powers 2^-n)", True, "rows_h", "exact"),
        ("contiguous, shard clocks on the h grid (lam^dT = 2^-n), rows free", True, "shards_h", "exact"),
        ("interleaved, own clocks, register * lam^dT rounded once to the grid", False, "free", "exact"),
        ("interleaved, shard clocks on the h grid (lam^dT = 2^-n), rows free", False, "shards_h", "exact"),
        ("interleaved, own clocks, plain float sums * lam^dT", False, "free", "float"),
        ("contiguous, shard clocks on the h grid, power split by whole half-lives", True, "shards_h", "exact+split"),
        ("interleaved, shard clocks on the h grid, power split by whole half-lives", False, "shards_h", "exact+split"),
    ]
    for k in (2, 5):
        n_fields = 1 + k + k * (k + 1) // 2
        print(f" k = {k} ({n_fields} fields):")
        results = {c[0]: [0, 0, 0, 0] for c in cases}
        stream = [0, 0, 0]
        for ci, (name, contiguous, clock, mode) in enumerate(cases):
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
                ref = [a.round_to_f64() for a in decayed_kulisch(terms, ts, range(N_ROWS), T_end, h)]
                if ci == 0:
                    sf = streaming_float(terms, ts, h)
                    d = [ulps(a, b) for a, b in zip(sf, ref)]
                    stream[0] += any(d)
                    stream[1] += sum(v > 0 for v in d)
                    stream[2] = max(stream[2], max(d))
                for _ in range(n_splits):
                    parts = make_split(N_ROWS, srng, contiguous)
                    if mode == "common":
                        st = [decayed_kulisch(terms, ts, p, T_end, h) for p in parts]
                        merged = random_tree(st, kulisch_merge, srng)
                        out = [a.round_to_f64() for a in merged]
                    else:
                        # Each shard in its own clock: its last row's time, or
                        # the first point at or after it on a grid of h back
                        # from T_end, where every offset's power is 2^-n.
                        def own(p, T_end=T_end, clock=clock):
                            last = ts[p[-1]]
                            return T_end - h * ((T_end - last) // h) if clock == "shards_h" else last

                        if mode == "float":
                            st = [(own(p), decayed_float(terms, ts, p, own(p), h)) for p in parts]

                            def comb(a, b):
                                (Ta, Sa), (Tb, Sb) = sorted([a, b], key=lambda s: s[0])
                                f = pow_lam(Tb - Ta, h)
                                return Tb, [f * u + v for u, v in zip(Sa, Sb)]
                        else:
                            st = [(own(p), decayed_kulisch(terms, ts, p, own(p), h)) for p in parts]

                            def comb(a, b, mode=mode):
                                (Ta, Sa), (Tb, Sb) = sorted([a, b], key=lambda s: s[0])
                                f = pow_lam(Tb - Ta, h)
                                for ua, ub in zip(Sa, Sb):
                                    if mode == "round":
                                        ub.add(f * ua.round_to_f64())
                                    else:
                                        ub.merge(scale_exact_then_round(ua, f))
                                return Tb, Sb

                        Tm, merged = random_tree(st, comb, srng)
                        assert Tm == T_end
                        out = [a if isinstance(a, float) else a.round_to_f64() for a in merged]
                    d = [ulps(a, b) for a, b in zip(out, ref)]
                    r = results[name]
                    r[0] += any(d)
                    r[1] += sum(v > 0 for v in d)
                    r[2] = max(r[2], max(d))
                    r[3] += 1
            r = results[name]
            print(
                f"   {name:<72} splits differing {r[0]:3d} / {r[3]}; fields {r[1]:4d} / {r[3] * n_fields}; worst {r[2]} ulps"
            )
        SPLIT_POW = False
        print(
            f"   {'unsharded streaming recursion S <- lam^dt S + term (floats)':<72} sets differing {stream[0]:3d} / {n_sets}; fields {stream[1]:4d} / {n_sets * n_fields}; worst {stream[2]} ulps"
        )


# --------------------------------------------------------------------------
# 3. Operations and bytes per merge
# --------------------------------------------------------------------------


def report_ops() -> None:
    print()
    print("== 5. Operations and bytes per merge, per scalar state field ==")
    print("   plain float, no decay  : 1 add; 8 B")
    print("   plain float, decay     : 1 multiply + 1 add (one shared pow per merge); 8 B")
    print(f"   Kulisch, no decay      : {K_LIMBS} limb adds with carry + 1 flag OR; {8 * K_LIMBS + 1} B")
    print(
        f"   Kulisch, decay (exact) : {K_LIMBS} limb x 53-bit multiplies + {K_LIMBS} shifts (one rounding) + {K_LIMBS} limb adds; {8 * K_LIMBS + 1} B"
    )
    print(
        f"   Kulisch, decay (round) : {K_LIMBS}-limb scan to round + 1 multiply + a two-limb add and its carry ripple; {8 * K_LIMBS + 1} B"
    )
    print(
        f"   Kulisch, power-of-two decay: a shift by n bits of {K_LIMBS} limbs ({K_LIMBS} shifts + ORs) + {K_LIMBS} limb adds; {8 * K_LIMBS + 1} B"
    )
    for k in (2, 5, 20):
        n_fields = 1 + k + k * (k + 1) // 2
        print(
            f"   k = {k:>2}: {n_fields:3d} summed fields (W, S1, S2) -> float {8 * n_fields:6d} B, {n_fields:4d} adds;"
            f" Kulisch {(8 * K_LIMBS + 1) * n_fields:6d} B, {K_LIMBS * n_fields:5d} limb adds"
        )


def main() -> None:
    quick = "--quick" in sys.argv
    t0 = time.time()
    check_no_decay(3 if quick else 8, 4 if quick else 12)
    check_centering()
    check_decay(2 if quick else 4, 3 if quick else 8)
    report_ops()
    print(f"\n(total {time.time() - t0:.1f} s, seed {SEED}, Python {sys.version.split()[0]}, numpy {np.__version__})")


if __name__ == "__main__":
    main()
