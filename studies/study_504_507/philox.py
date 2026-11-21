"""Philox counter-based generators, written from Salmon et al. (2011), and the
draw stream the study proposes for rheon's state.

Salmon, Moraes, Dror and Shaw, "Parallel random numbers: as easy as 1, 2, 3",
SC'11. Philox-NxW-R is a keyed bijection of an N-word counter: R rounds, each
an S-box of two W-bit multiplications (keeping the high and low halves of each
product) followed by a fixed permutation of the words, with the key bumped by
a Weyl constant between rounds. Output block = f(counter, key); nothing else
is state, so draw ``j`` of row ``n`` is a pure function of its coordinates.

Philox4x64-10 (one round, words ``(c0, c1, c2, c3)``, key ``(k0, k1)``)::

    (hi0, lo0) = M0 * c0          (hi1, lo1) = M1 * c2     (128-bit products)
    c' = (hi1 ^ c1 ^ k0,  lo1,  hi0 ^ c3 ^ k1,  lo0)
    k' = (k0 + W0, k1 + W1)  mod 2^64                      (between rounds)

with M0 = 0xD2E7470EE14C6C93, M1 = 0xCA5A826395121157, W0 = 0x9E3779B97F4A7C15
(the golden ratio), W1 = 0xBB67AE8584CAA73B (sqrt(3) - 1). Philox4x32-10 is the
same with 32-bit words, M0 = 0xD2511F53, M1 = 0xCD9E8D57, W0 = 0x9E3779B9,
W1 = 0xBB67AE85. numpy's ``numpy.random.Philox`` is Philox4x64-10.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

M64 = (1 << 64) - 1
M32 = (1 << 32) - 1
PH4x64_M0, PH4x64_M1 = 0xD2E7470EE14C6C93, 0xCA5A826395121157
PH4x64_W0, PH4x64_W1 = 0x9E3779B97F4A7C15, 0xBB67AE8584CAA73B
PH4x32_M0, PH4x32_M1 = 0xD2511F53, 0xCD9E8D57
PH4x32_W0, PH4x32_W1 = 0x9E3779B9, 0xBB67AE85
ROUNDS = 10


def philox4x64(
    ctr: tuple[int, int, int, int], key: tuple[int, int], rounds: int = ROUNDS
) -> tuple[int, ...]:
    """One Philox4x64 block, scalar, from the paper's definition."""
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in range(rounds):
        if r:
            k0 = (k0 + PH4x64_W0) & M64
            k1 = (k1 + PH4x64_W1) & M64
        p0 = PH4x64_M0 * c0
        p1 = PH4x64_M1 * c2
        c0, c1, c2, c3 = ((p1 >> 64) ^ c1 ^ k0, p1 & M64, (p0 >> 64) ^ c3 ^ k1, p0 & M64)
    return c0, c1, c2, c3


def philox4x32(
    ctr: tuple[int, int, int, int], key: tuple[int, int], rounds: int = ROUNDS
) -> tuple[int, ...]:
    """One Philox4x32 block, scalar (not in numpy: checked by the paper's known answers)."""
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in range(rounds):
        if r:
            k0 = (k0 + PH4x32_W0) & M32
            k1 = (k1 + PH4x32_W1) & M32
        p0 = PH4x32_M0 * c0
        p1 = PH4x32_M1 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0, p1 & M32, (p0 >> 32) ^ c3 ^ k1, p0 & M32)
    return c0, c1, c2, c3


# Known-answer vectors of Random123's kat_vectors (counter words, key words ->
# output words), for the three standard inputs: zeros, all ones, and the
# digits of pi.
KAT_4x32 = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((M32, M32, M32, M32), (M32, M32), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]
KAT_4x64 = [
    (
        (0, 0, 0, 0),
        (0, 0),
        (0x16554D9ECA36314C, 0xDB20FE9D672D0FDC, 0xD7E772CEE186176B, 0x7E68B68AEC7BA23B),
    ),
    (
        (M64, M64, M64, M64),
        (M64, M64),
        (0x87B092C3013FE90B, 0x438C3C67BE8D0224, 0x9CC7D7C69CD777B6, 0xA09CAEBF594F0BA0),
    ),
    (
        (0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89),
        (0x452821E638D01377, 0xBE5466CF34E90C6C),
        (0xA528F45403E61D95, 0x38C72DBD566E9788, 0xA5A1610E72FD18B5, 0x57BD43B5E52B7FE6),
    ),
]


def _mulhilo64(a: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    """128-bit product of uint64 arrays by a constant, as (hi, lo), in 32-bit halves."""
    m = np.uint64(M32)
    s = np.uint64(32)
    a_lo, a_hi = a & m, a >> s
    b_lo, b_hi = np.uint64(b & M32), np.uint64(b >> 32)
    ll = a_lo * b_lo
    lh = a_lo * b_hi
    hl = a_hi * b_lo
    hh = a_hi * b_hi
    mid = (ll >> s) + (lh & m) + (hl & m)
    lo = (ll & m) | ((mid & m) << s)
    hi = hh + (lh >> s) + (hl >> s) + (mid >> s)
    return hi, lo


def philox4x64_np(ctr: np.ndarray, key: tuple[int, int], rounds: int = ROUNDS) -> np.ndarray:
    """Vectorised Philox4x64: ``ctr`` is (n, 4) uint64, the result (n, 4) uint64."""
    c = [ctr[:, i].astype(np.uint64) for i in range(4)]
    k0, k1 = key
    with np.errstate(over="ignore"):
        for r in range(rounds):
            if r:
                k0 = (k0 + PH4x64_W0) & M64
                k1 = (k1 + PH4x64_W1) & M64
            hi0, lo0 = _mulhilo64(c[0], PH4x64_M0)
            hi1, lo1 = _mulhilo64(c[2], PH4x64_M1)
            c = [hi1 ^ c[1] ^ np.uint64(k0), lo1, hi0 ^ c[3] ^ np.uint64(k1), lo0]
    return np.stack(c, axis=1)


# --- numpy's Philox, driven to a given block -------------------------------------


def numpy_block(ctr: tuple[int, int, int, int], key: tuple[int, int]) -> tuple[int, ...]:
    """The block numpy's Philox computes at counter ``ctr`` under ``key``.

    numpy's ``philox_next`` adds one to the 256-bit counter (word 0 least
    significant, carry into 1, 2, 3) *before* computing a block, then serves the
    block's four words in order. So the state is set one counter below ``ctr``
    with the buffer empty (``buffer_pos = 4``), and four raw words are read.
    """
    c = sum(w << (64 * i) for i, w in enumerate(ctr))
    c = (c - 1) % (1 << 256)
    bg = np.random.Philox(key=0)
    st = bg.state
    st["state"]["counter"] = np.array([(c >> (64 * i)) & M64 for i in range(4)], dtype=np.uint64)
    st["state"]["key"] = np.array(key, dtype=np.uint64)
    st["buffer"] = np.zeros(4, dtype=np.uint64)
    st["buffer_pos"] = 4
    st["has_uint32"] = 0
    st["uinteger"] = 0
    bg.state = st
    return tuple(int(w) for w in bg.random_raw(4))


def numpy_uniforms(ctr: tuple[int, int, int, int], key: tuple[int, int], n: int) -> np.ndarray:
    """``Generator(Philox).random(n)`` from block ``ctr`` on."""
    c = sum(w << (64 * i) for i, w in enumerate(ctr))
    c = (c - 1) % (1 << 256)
    bg = np.random.Philox(key=0)
    st = bg.state
    st["state"]["counter"] = np.array([(c >> (64 * i)) & M64 for i in range(4)], dtype=np.uint64)
    st["state"]["key"] = np.array(key, dtype=np.uint64)
    st["buffer"] = np.zeros(4, dtype=np.uint64)
    st["buffer_pos"] = 4
    bg.state = st
    return np.random.Generator(bg).random(n)


# --- word -> number conversions ----------------------------------------------------

TWO_M53 = 2.0**-53


def u01(w: int) -> float:
    """[0, 1): the top 53 bits times 2^-53 (numpy's ``next_double``); exact."""
    return (w >> 11) * TWO_M53


def u01_open(w: int) -> float:
    """(0, 1): the top 52 bits with a 1 appended, times 2^-53; an odd multiple
    of 2^-53, so neither 0 nor 1 and exact (one more bit would need rounding)."""
    return ((w >> 11) | 1) * TWO_M53


def below(w: int, k: int) -> int:
    """An index in [0, k): the high 64 bits of the 128-bit product ``w * k``
    (Lemire's multiply-shift, no rejection; bias at most k / 2^64). Integer only."""
    return (w * k) >> 64


# Wichura (1988), Algorithm AS 241, PPND16: the normal quantile to about 1e-16.
_A = [
    3.3871328727963666080e0,
    1.3314166789178437745e2,
    1.9715909503065514427e3,
    1.3731693765509461125e4,
    4.5921953931549871457e4,
    6.7265770927008700853e4,
    3.3430575583588128105e4,
    2.5090809287301226727e3,
]
_B = [
    1.0,
    4.2313330701600911252e1,
    6.8718700749205790830e2,
    5.3941960214247511077e3,
    2.1213794301586595867e4,
    3.9307895800092710610e4,
    2.8729085735721942674e4,
    5.2264952788528545610e3,
]
_C = [
    1.42343711074968357734e0,
    4.63033784615654529590e0,
    5.76949722146069140550e0,
    3.64784832476320460504e0,
    1.27045825245236838258e0,
    2.41780725177450611770e-1,
    2.27238449892691845833e-2,
    7.74545014278341407640e-4,
]
_D = [
    1.0,
    2.05319162663775882187e0,
    1.67638483018380384940e0,
    6.89767334985100004550e-1,
    1.48103976427480074590e-1,
    1.51986665636164571966e-2,
    5.47593808499534494600e-4,
    1.05075007164441684324e-9,
]
_E = [
    6.65790464350110377720e0,
    5.46378491116411436990e0,
    1.78482653991729133580e0,
    2.96560571828504891230e-1,
    2.65321895265761230930e-2,
    1.24266094738807843860e-3,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
]
_F = [
    1.0,
    5.99832206555887937690e-1,
    1.36929880922735805310e-1,
    1.48753612908506148525e-2,
    7.86869131145613259100e-4,
    1.84631831751005468180e-5,
    1.42151175831644588870e-7,
    2.04426310338993978564e-15,
]


def _horner(c: list[float], x: float) -> float:
    s = c[7]
    for a in reversed(c[:7]):
        s = s * x + a
    return s


def ppnd16(p: float) -> float:
    """The standard normal quantile of ``p`` in (0, 1): + - * / on 85% of draws,
    one ``ln`` and one ``sqrt`` on the tails (the ``ln`` is the platform's)."""
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * _horner(_A, r) / _horner(_B, r)
    r = p if q < 0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        v = _horner(_C, r) / _horner(_D, r)
    else:
        r -= 5.0
        v = _horner(_E, r) / _horner(_F, r)
    return -v if q < 0 else v


# --- the group hash: rheon's FixedHasher, ported ------------------------------------

_FH_MUL = 0x517CC1B727220A95


class FixedHasher:
    """``crates/rheon-core/src/hasher.rs``: multiply-and-rotate over little-endian
    64-bit words, SplitMix64's finalizer. The hash rheon's group maps already use,
    the same in every process on every platform."""

    def __init__(self) -> None:
        self.h = 0

    def _add(self, word: int) -> None:
        h = self.h
        h = ((h << 5) | (h >> 59)) & M64
        self.h = ((h ^ word) * _FH_MUL) & M64

    def write(self, b: bytes) -> None:
        n8 = len(b) // 8 * 8
        for i in range(0, n8, 8):
            self._add(int.from_bytes(b[i : i + 8], "little"))
        rest = b[n8:]
        if rest:
            buf = bytearray(8)
            buf[: len(rest)] = rest
            buf[7] = len(rest)
            self._add(int.from_bytes(bytes(buf), "little"))

    def write_u8(self, i: int) -> None:
        self._add(i)

    def write_u64(self, i: int) -> None:
        self._add(i & M64)

    def write_str(self, s: str) -> None:
        # Rust's `impl Hash for str`: the bytes, then a 0xff terminator.
        self.write(s.encode("utf-8"))
        self.write_u8(0xFF)

    def finish(self) -> int:
        z = (self.h + 0x9E3779B97F4A7C15) & M64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


def group_hash(key: object) -> int:
    """A group key's 64-bit hash: a string as Rust hashes a ``str``, an integer
    as a ``u64``, a tuple as its parts in order (Rust's tuple ``Hash``)."""
    h = FixedHasher()

    def feed(k: object) -> None:
        if isinstance(k, tuple):
            for p in k:
                feed(p)
        elif isinstance(k, str):
            h.write_str(k)
        elif isinstance(k, int):
            h.write_u64(k)
        else:
            raise TypeError(type(k))

    feed(key)
    return h.finish()


# --- the draw stream ---------------------------------------------------------------

DEFAULT_SEED = 0


@dataclass(frozen=True)
class DrawStream:
    """The proposed layout (Philox4x64-10; numpy's generator)::

        key     = (seed, group_hash)                   two u64
        counter = (j, lane, n, slot)                    four u64
          j     the block index inside one row's draws (four u64 words a block)
          lane  what the draws are for: 0 the per-row policy draws, 1 a model's
                construction (half-space trees), 2 a cut on insertion (random cut
                forest), ... a frozen small integer, like ModelState's tags
          n     the learned-row count *before* this row (rows of weight > 0 the
                model has learned); the one number the state keeps for the stream
          slot  the model's slot in the bank (0 for one model)

    Word ``i`` of a row's draws is word ``i % 4`` of block ``(i // 4, lane, n, slot)``.
    """

    seed: int = DEFAULT_SEED
    group: int = 0
    lane: int = 0
    slot: int = 0

    @property
    def key(self) -> tuple[int, int]:
        return (self.seed & M64, self.group & M64)

    def words(self, n: int, count: int, lane: int | None = None) -> list[int]:
        ln = self.lane if lane is None else lane
        out: list[int] = []
        j = 0
        while len(out) < count:
            out.extend(philox4x64((j, ln, n, self.slot), self.key))
            j += 1
        return out[:count]

    def words_np(self, n: int, count: int, lane: int | None = None) -> np.ndarray:
        ln = self.lane if lane is None else lane
        nb = (count + 3) // 4
        ctr = np.zeros((nb, 4), dtype=np.uint64)
        ctr[:, 0] = np.arange(nb, dtype=np.uint64)
        ctr[:, 1] = ln
        ctr[:, 2] = n
        ctr[:, 3] = self.slot
        return philox4x64_np(ctr, self.key).reshape(-1)[:count]

    def uniforms(self, n: int, count: int, lane: int | None = None) -> np.ndarray:
        w = self.words_np(n, count, lane)
        return (w >> np.uint64(11)).astype(np.float64) * TWO_M53

    def normals(self, n: int, count: int, lane: int | None = None) -> np.ndarray:
        w = self.words_np(n, count, lane)
        u = ((w >> np.uint64(11)) | np.uint64(1)).astype(np.float64) * TWO_M53
        return np.array([ppnd16(float(x)) for x in u])


def f64_bytes(a: np.ndarray | list[float] | float) -> bytes:
    a = np.atleast_1d(np.asarray(a, dtype=np.float64))
    return struct.pack(f"<{a.size}d", *a.tolist())
