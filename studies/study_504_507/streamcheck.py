"""The generator's checks: numpy to the bit, the paper's known answers, and the
draws' invariance to chunking, save/resume and the process.

The consumer here is the smallest model that draws: per group it keeps the
learned-row count ``n`` and nothing else; every row emits draws keyed on
``(seed, group, n)`` (three uniforms, one index in [0, 5), one normal), and a
row of weight > 0 then learns (``n += 1``). A zero-weight row emits the same
draws as the next learned row of its group and does not move ``n``.
"""

from __future__ import annotations

import hashlib
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import philox as P
import scipy.special as sps


def check_kats() -> list[tuple[str, int, int]]:
    rows = []
    ok64 = sum(P.philox4x64(c, k) == o for c, k, o in P.KAT_4x64)
    ok64n = sum(P.numpy_block(c, k) == o for c, k, o in P.KAT_4x64)
    ok32 = sum(P.philox4x32(c, k) == o for c, k, o in P.KAT_4x32)
    rows.append(("Philox4x64-10, longhand vs Random123 known answers", ok64, len(P.KAT_4x64)))
    rows.append(("Philox4x64-10, numpy vs Random123 known answers", ok64n, len(P.KAT_4x64)))
    rows.append(("Philox4x32-10, longhand vs Random123 known answers", ok32, len(P.KAT_4x32)))
    return rows


def check_numpy(n_random: int = 3000, seed: int = 504) -> list[tuple[str, int, int]]:
    """Blocks at random counters and keys, and the layout's own counters, against
    numpy's Philox; uniforms against ``Generator(Philox).random``."""
    rng = random.Random(seed)
    rows = []
    bad = 0
    for _ in range(n_random):
        c = tuple(rng.getrandbits(64) for _ in range(4))
        k = (rng.getrandbits(64), rng.getrandbits(64))
        bad += P.philox4x64(c, k) != P.numpy_block(c, k)
    rows.append(
        ("random 256-bit counters and 128-bit keys: blocks equal", n_random - bad, n_random)
    )
    # Edge counters: numpy's carry (counter - 1 across a word boundary).
    edges = [
        (0, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (P.M64, P.M64, P.M64, P.M64),
        (0, 0, 2**63, 7),
    ]
    bad = sum(P.philox4x64(c, (11, 13)) != P.numpy_block(c, (11, 13)) for c in edges)
    rows.append(
        ("counters at word boundaries (numpy's carry): blocks equal", len(edges) - bad, len(edges))
    )
    # The layout's counters: (j, lane, n, slot) for a few groups and rows.
    cases = 0
    bad = 0
    for g in ["a", "group-17", ("x", 3), 12345]:
        s = P.DrawStream(seed=20261121, group=P.group_hash(g), lane=0, slot=0)
        for n in [0, 1, 2, 1000, 2**32, 2**40 + 3]:
            for j in range(3):
                cases += 1
                bad += P.philox4x64((j, 0, n, 0), s.key) != P.numpy_block((j, 0, n, 0), s.key)
    rows.append(("the layout's counters (j, lane, n, slot): blocks equal", cases - bad, cases))
    # Uniforms: the vectorised words -> [0, 1) against numpy's Generator.random.
    s = P.DrawStream(seed=7, group=P.group_hash("g"))
    bad = 0
    cases = 0
    for n in [0, 5, 99]:
        mine = s.uniforms(n, 4)
        ref = P.numpy_uniforms((0, 0, n, 0), s.key, 4)
        cases += 4
        bad += int(np.sum(mine.view(np.uint64) != ref.view(np.uint64)))
    rows.append(
        ("uniforms [0,1) of a row (one block) vs Generator(Philox).random", cases - bad, cases)
    )
    # Vectorised Philox against the scalar one.
    c = np.array([[rng.getrandbits(64) for _ in range(4)] for _ in range(1000)], dtype=np.uint64)
    v = P.philox4x64_np(c, (3, 5))
    ok = sum(
        tuple(int(x) for x in v[i]) == P.philox4x64(tuple(int(x) for x in c[i]), (3, 5))
        for i in range(1000)
    )
    rows.append(("vectorised numpy form vs scalar longhand", ok, 1000))
    return rows


def check_normals(count: int = 20000) -> dict[str, float]:
    s = P.DrawStream(seed=1, group=2)
    w = s.words_np(0, count)
    p = ((w >> np.uint64(11)) | np.uint64(1)).astype(np.float64) * P.TWO_M53
    z = np.array([P.ppnd16(float(x)) for x in p])
    r = sps.ndtri(p)
    ulp = np.abs(z - r) / np.spacing(np.abs(r))
    tails = np.abs(p - 0.5) > 0.425
    import mpmath as mp

    mp.mp.prec = 200
    m = 3000
    ex = np.array([float(mp.sqrt(2) * mp.erfinv(2 * mp.mpf(float(x)) - 1)) for x in p[:m]])
    e_pp = np.abs(z[:m] - ex) / np.spacing(np.abs(ex))
    e_nd = np.abs(r[:m] - ex) / np.spacing(np.abs(ex))
    return {
        "max ulps vs scipy ndtri": float(ulp.max()),
        "mean ulps vs scipy ndtri": float(ulp.mean()),
        "share on the + - * / branch": float(1 - tails.mean()),
        "max ulps vs ndtri on that branch": float(ulp[~tails].max()),
        f"max ulps vs exact (mpmath, {m} draws)": float(e_pp.max()),
        "mean ulps vs exact": float(e_pp.mean()),
        "ndtri's own max ulps vs exact": float(e_nd.max()),
        "ndtri's own mean ulps vs exact": float(e_nd.mean()),
    }


# --- the invariance harness --------------------------------------------------------


def make_rows(n_rows: int, n_groups: int, seed: int) -> list[tuple[str, float]]:
    rng = np.random.default_rng(seed)
    groups = [f"g{int(i)}" for i in rng.integers(0, n_groups, n_rows)]
    w = rng.choice([0.0, 1.0, 0.5, 2.0], size=n_rows, p=[0.2, 0.6, 0.1, 0.1])
    w[0] = 0.0  # the first row of the stream has weight zero
    # and the first row of one group is a run of zero-weight rows
    first = {g: i for i, g in reversed(list(enumerate(groups)))}
    for i in list(first.values())[:3]:
        w[i] = 0.0
    return list(zip(groups, w.tolist(), strict=True))


class Consumer:
    def __init__(self, seed: int = P.DEFAULT_SEED, hash_fn=P.group_hash) -> None:
        self.seed = seed
        self.hash_fn = hash_fn
        self.n: dict[str, int] = {}

    def save(self) -> bytes:
        out = struct.pack("<Q", self.seed)
        for g in sorted(self.n):
            b = g.encode()
            out += struct.pack("<H", len(b)) + b + struct.pack("<Q", self.n[g])
        return out

    @classmethod
    def load(cls, b: bytes) -> Consumer:
        (seed,) = struct.unpack_from("<Q", b, 0)
        c = cls(seed)
        i = 8
        while i < len(b):
            (ln,) = struct.unpack_from("<H", b, i)
            g = b[i + 2 : i + 2 + ln].decode()
            (n,) = struct.unpack_from("<Q", b, i + 2 + ln)
            c.n[g] = n
            i += 2 + ln + 8
        return c

    def feed(self, rows: list[tuple[str, float]]) -> bytes:
        out = bytearray()
        for g, w in rows:
            n = self.n.get(g, 0)
            s = P.DrawStream(self.seed, self.hash_fn(g), lane=0)
            words = s.words(n, 5)
            u = [P.u01(x) for x in words[:3]]
            k = P.below(words[3], 5)
            z = P.ppnd16(P.u01_open(words[4]))
            out += struct.pack("<3dQd", *u, k, z)
            if w > 0:
                self.n[g] = n + 1
        return bytes(out)


def run_chunked(
    rows, chunks: list[int], resume_every: int | None = None, seed: int = P.DEFAULT_SEED
) -> str:
    c = Consumer(seed)
    h = hashlib.sha256()
    i = 0
    for t, size in enumerate(chunks):
        h.update(c.feed(rows[i : i + size]))
        i += size
        if resume_every and (t + 1) % resume_every == 0:
            c = Consumer.load(c.save())
    assert i == len(rows)
    return h.hexdigest()[:16]


def chunkings(n: int, seed: int = 9) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    rand = []
    left = n
    while left:
        s = int(min(left, rng.integers(1, 300)))
        rand.append(s)
        left -= s
    return {
        "1 chunk": [n],
        f"{n} chunks of 1 row": [1] * n,
        "1000 chunks": [n // 1000 + (1 if i < n % 1000 else 0) for i in range(1000)],
        f"{len(rand)} random chunks": rand,
    }


def check_invariance(n_rows: int = 5000) -> list[tuple[str, str]]:
    rows = make_rows(n_rows, 40, 504)
    out = []
    for name, ch in chunkings(n_rows).items():
        out.append((name, run_chunked(rows, ch)))
        out.append((name + ", save/load after every chunk", run_chunked(rows, ch, resume_every=1)))
    return out


def digest_in_process(n_rows: int = 2000) -> str:
    rows = make_rows(n_rows, 40, 504)
    return run_chunked(rows, [n_rows])


def python_hash_digest(n_rows: int = 500) -> str:
    """The same consumer keyed on Python's own ``hash(str)``: salted per process."""
    rows = make_rows(n_rows, 40, 504)
    c = Consumer(hash_fn=lambda g: hash(g) & P.M64)
    return hashlib.sha256(c.feed(rows)).hexdigest()[:16]


def check_processes() -> list[tuple[str, str, str]]:
    here = Path(__file__).resolve().parent
    code = (
        f"import sys; sys.path.insert(0, {str(here)!r}); import streamcheck as S;"
        "print(S.digest_in_process(), S.python_hash_digest())"
    )
    res = []
    for hs in ["0", "4242", "random"]:
        env = dict(os.environ, PYTHONHASHSEED=hs)
        o = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        a, b = o.stdout.split()
        res.append((f"PYTHONHASHSEED={hs}", a, b))
    return res


def check_zero_weight() -> list[tuple[str, bool]]:
    """Rule 9 for the stream: the zero-weight first row emits the draws of the
    first learned row and leaves ``n`` at 0."""
    c = Consumer()
    a = c.feed([("g", 0.0)])
    n_after_zero = c.n.get("g", 0)
    b = c.feed([("g", 1.0)])
    d = c.feed([("g", 0.0), ("g", 0.0)])
    e = c.feed([("g", 1.0)])
    return [
        ("a zero-weight first row leaves n at 0", n_after_zero == 0),
        ("it emits the draws of the first learned row (n = 0)", a == b),
        (
            "zero-weight rows after one learned row emit the n = 1 draws",
            d[: len(e)] == e and d[len(e) :] == e,
        ),
        ("n counts learned rows only (1 + 1 = 2)", c.n["g"] == 2),
    ]
