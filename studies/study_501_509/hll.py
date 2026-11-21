"""HyperLogLog over a column's values, as ``audit`` would key them.

**Keys.** ``audit`` keys a value by its f64 bits with ``-0`` read as ``0``
(``audit.rs``, ``key``); a NaN keeps its payload, so the null the plumbing
writes (``0x7FF8_0000_004E_554C``) counts apart from a NaN in the data. The
sketch takes the same key, so below ``distinct_cap`` (exact) and past it
(this sketch) the count is of the same thing.

**Hash.** ``h = splitmix64_finalizer(key)``: ``audit``'s ``KeyHasher::write_u64``
already, constants fixed in the source, no seed drawn at run time::

    z = key + 0x9e3779b97f4a7c15               (mod 2^64)
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb
    h = z ^ (z >> 31)

A bijection of 64-bit words, so two distinct keys never share a hash.

**Registers** (datasketch's layout, so its classes take this hash and give
the same array): ``m = 2^p`` registers of one byte; a value goes to register
``j = h & (m - 1)`` and offers ``rho = (B - p) - bitlen(h >> p) + 1``, the
position of the leftmost 1 in the ``B - p`` bits left (``B = 64``; ``B = 32``
for datasketch's plain ``HyperLogLog``, fed ``h``'s top 32 bits); the
register keeps the largest offer. Order and chunking cannot change a maximum,
and two sketches merge by the register-wise maximum, which is the sketch of
the union exactly.

**Estimate** (Flajolet et al. 2007): ``E = alpha_m m^2 / sum_j 2^-M_j``;
under ``2.5 m`` linear counting ``m ln(m / V)`` (``V`` the empty registers)
while ``V > 0``. No large-range correction at 64 bits. HLL++ (Heule et al.
2013, dense form, datasketch's ``HyperLogLogPlusPlus.count``): linear counting
while it is under the precision's threshold, else ``E`` minus the bias
interpolated from the paper's tables (mean of the six nearest raw estimates)
while ``E <= 5 m``. The relative standard error is about ``1.04 / sqrt(m)``.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

M64 = (1 << 64) - 1
NULL_BITS = 0x7FF8_0000_004E_554C


def keys(values: np.ndarray) -> np.ndarray:
    """``audit``'s key: the f64 bits, ``-0.0`` read as ``0`` (``+0.0``'s bits)."""
    v = np.ascontiguousarray(values, dtype=np.float64)
    k = v.view(np.uint64).copy()
    k[v == 0.0] = 0
    return k


def splitmix(k: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = k.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def splitmix_int(k: int) -> int:
    z = (k + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def fixed_hasher_u64(k: np.ndarray) -> np.ndarray:
    """``hasher.rs``'s ``FixedHasher`` after one ``write_u64``: the group
    keys' hasher, the other candidate in the source."""
    with np.errstate(over="ignore"):
        s = k.astype(np.uint64) * np.uint64(0x517CC1B727220A95)  # (rotl(0, 5) ^ k) * MUL
        return splitmix(s)


def sha1_64(k: np.ndarray) -> np.ndarray:
    """datasketch's default ``sha1_hash64`` over the key's 8 little-endian
    bytes: the first 8 bytes of SHA-1, read little-endian."""
    out = np.empty(len(k), np.uint64)
    for i, x in enumerate(k.tolist()):
        out[i] = struct.unpack("<Q", hashlib.sha1(struct.pack("<Q", x)).digest()[:8])[0]
    return out


def identity(k: np.ndarray) -> np.ndarray:
    """No hash at all: the check that the avalanche and error tables bite."""
    return k.astype(np.uint64)


def bitlen(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64).copy()
    n = np.zeros(x.shape, np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        big = (x >> np.uint64(s)) != 0
        n[big] += s
        x[big] >>= np.uint64(s)
    n += (x != 0).astype(np.int64)
    return n


class HLL:
    def __init__(self, p: int, bits: int = 64, hashf=splitmix):
        if not 4 <= p <= 18:
            raise ValueError("p from 4 to 18")
        self.p, self.m, self.bits, self.hashf = p, 1 << p, bits, hashf
        self.reg = np.zeros(self.m, np.int8)

    def hashes(self, values: np.ndarray) -> np.ndarray:
        h = self.hashf(keys(values))
        return h >> np.uint64(32) if self.bits == 32 else h

    def update(self, values: np.ndarray, weights: np.ndarray | None = None,
               count_zero_weight: bool = True) -> None:
        values = np.asarray(values, np.float64)
        if weights is not None and not count_zero_weight:
            values = values[np.asarray(weights) > 0.0]
        h = self.hashes(values)
        j = (h & np.uint64(self.m - 1)).astype(np.int64)
        rho = (self.bits - self.p) - bitlen(h >> np.uint64(self.p)) + 1
        np.maximum.at(self.reg, j, rho.astype(np.int8))

    def merge(self, other: HLL) -> HLL:
        out = HLL(self.p, self.bits, self.hashf)
        out.reg = np.maximum(self.reg, other.reg)
        return out

    def alpha(self) -> float:
        return {16: 0.673, 32: 0.697, 64: 0.709}.get(self.m, 0.7213 / (1.0 + 1.079 / self.m))

    def raw(self) -> float:
        return self.alpha() * float(self.m**2) / float(np.sum(2.0 ** (-self.reg.astype(float))))

    def count(self) -> float:
        """Plain HLL (64-bit: no large-range correction)."""
        e = self.raw()
        if e <= 2.5 * self.m:
            zeros = self.m - int(np.count_nonzero(self.reg))
            if zeros > 0:
                return self.m * float(np.log(self.m / zeros))
        return e

    def count_pp(self) -> float:
        """HLL++'s dense estimate with datasketch's bias tables (the paper's)."""
        from datasketch import hyperloglog_const as c

        zeros = self.m - int(np.count_nonzero(self.reg))
        if zeros > 0:
            lc = self.m * float(np.log(self.m / float(zeros)))
            if lc <= c._thresholds[self.p - 4]:
                return lc
        e = self.raw()
        if e <= 5 * self.m:
            bias = np.asarray(c._bias[self.p - 4])
            est = np.asarray(c._raw_estimate[self.p - 4])
            nn = np.argsort((e - est) ** 2)[:6]
            return e - float(np.mean(bias[nn]))
        return e

    def nbytes(self) -> int:
        return self.m
