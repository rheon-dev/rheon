"""The HyperLogLog checks (task 509). Each returns (header, rows) for a table."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

import hll as H
import numpy as np
from datasketch import HyperLogLog, HyperLogLogPlusPlus

PRECISIONS = (4, 6, 8, 10, 12, 14)


def values(kind: str, n: int, trial: int) -> np.ndarray:
    """Distinct values of a trial (no seed is drawn by the sketch: the trials
    differ by their data)."""
    if kind == "random":
        return np.random.default_rng(trial).normal(size=n) * 1e3
    if kind == "sequential":  # integers as floats, the scan's worry
        return np.arange(trial * n, (trial + 1) * n, dtype=np.float64)
    if kind == "grid":  # prices on a cent grid
        return np.round(100.0 + 0.01 * np.arange(trial * n, (trial + 1) * n), 2)
    raise ValueError(kind)


def truth(v: np.ndarray) -> int:
    return len(np.unique(H.keys(v)))


def vs_datasketch(n: int = 50_000):
    """Registers entry for entry and count() against datasketch with this
    hash handed in (hashfunc=): HLL++ on the 64-bit hash, plain HyperLogLog
    on its top 32 bits."""
    rows = []
    for p in (4, 8, 12, 14, 16):
        for kind in ("random", "sequential", "grid"):
            v = values(kind, n, 3)
            k = H.keys(v).tolist()
            ours = H.HLL(p)
            ours.update(v)
            ds = HyperLogLogPlusPlus(p=p, hashfunc=H.splitmix_int)
            for x in k:
                ds.update(x)
            ours32 = H.HLL(p, bits=32)
            ours32.update(v)
            ds32 = HyperLogLog(p=p, hashfunc=lambda x: H.splitmix_int(x) >> 32)
            for x in k:
                ds32.update(x)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c32 = ds32.count()
            rows.append((p, kind, n, bool(np.array_equal(ours.reg, ds.reg)),
                         abs(ours.count_pp() - ds.count()) / ds.count(),
                         bool(np.array_equal(ours32.reg, ds32.reg)),
                         abs(ours32.count() - c32) / c32))
    hdr = ("p", "values", "n", "regs = ds HLL++", "count rel gap (HLL++)",
           "regs = ds HLL (32-bit)", "count rel gap (HLL)")
    return hdr, rows


def default_hash(n: int = 20_000):
    """datasketch's own default (SHA-1) on the key's bytes: what rheon would
    need to match datasketch with no hashfunc handed in."""
    rows = []
    for p in (8, 12):
        v = values("random", n, 5)
        k = H.keys(v)
        ds = HyperLogLogPlusPlus(p=p)  # sha1_hash64
        for x in k.tolist():
            ds.update(x.to_bytes(8, "little"))
        ours = H.HLL(p, hashf=H.sha1_64)
        ours.update(v)
        mine = H.HLL(p)
        mine.update(v)
        rows.append((p, n, bool(np.array_equal(ours.reg, ds.reg)), ds.count(), ours.count_pp(),
                     mine.count_pp(), truth(v)))
    hdr = ("p", "n", "SHA-1 regs = ds default", "ds count", "ours, SHA-1", "ours, splitmix",
           "truth")
    return hdr, rows


def avalanche(n: int = 4_096):
    """Flip one input bit of each key; the share of output bits that flip,
    per (input bit, output bit). A good mixer gives 1/2 everywhere."""
    rows = []
    for kind in ("sequential", "grid", "random"):
        k = H.keys(values(kind, n, 1))
        for name, f in (("splitmix (audit KeyHasher)", H.splitmix),
                        ("FixedHasher (hasher.rs)", H.fixed_hasher_u64),
                        ("SHA-1 (datasketch default)", H.sha1_64), ("identity", H.identity)):
            base = f(k)
            worst = 0.0
            for b in range(64):
                flipped = f(k ^ np.uint64(1 << b)) ^ base
                share = np.array([((flipped >> np.uint64(o)) & np.uint64(1)).mean()
                                  for o in range(64)])
                worst = max(worst, float(np.max(np.abs(share - 0.5))))
            rows.append((kind, name, n, worst))
    return ("keys", "hash", "keys tested", "max abs (P(flip) - 1/2)"), rows


def error_band(trials: int = 200, mult: float = 20.0, kind: str = "random"):
    """The relative error of the estimate over trials at n = mult * m distinct
    values, against 1.04 / sqrt(m)."""
    rows = []
    for p in PRECISIONS:
        m = 1 << p
        n = int(mult * m)
        e_plain, e_pp = [], []
        for t in range(trials):
            v = values(kind, n, 1_000 + t)
            s = H.HLL(p)
            s.update(v)
            tr = truth(v)
            e_plain.append(s.count() / tr - 1)
            e_pp.append(s.count_pp() / tr - 1)
        e_plain, e_pp = np.array(e_plain), np.array(e_pp)
        se = 1.04 / np.sqrt(m)
        rows.append((p, m, n, trials, se, float(np.std(e_plain)), float(np.mean(e_plain)),
                     float(np.mean(np.abs(e_plain) <= 2 * se)),
                     float(np.mean(np.abs(e_plain) <= 3 * se)), float(np.std(e_pp)),
                     float(np.mean(e_pp))))
    hdr = ("p", "m", "n", "trials", "1.04/sqrt(m)", "sd (HLL)", "bias (HLL)", "within 2 SE",
           "within 3 SE", "sd (HLL++)", "bias (HLL++)")
    return hdr, rows


def small_range(p: int = 10, trials: int = 200):
    """Bias and spread across n at one precision: the transition from linear
    counting to the raw estimate (near 2.5 m), where plain HLL is biased and
    HLL++'s tables correct it."""
    m = 1 << p
    rows = []
    for mult in (0.1, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0):
        n = int(mult * m)
        a, b = [], []
        for t in range(trials):
            v = values("random", n, 50_000 + t)
            s = H.HLL(p)
            s.update(v)
            tr = truth(v)
            a.append(s.count() / tr - 1)
            b.append(s.count_pp() / tr - 1)
        rows.append((p, n, mult, float(np.mean(a)), float(np.std(a)), float(np.mean(b)),
                     float(np.std(b)), 1.04 / np.sqrt(m)))
    hdr = ("p", "n", "n/m", "bias (HLL)", "sd (HLL)", "bias (HLL++)", "sd (HLL++)", "1.04/sqrt(m)")
    return hdr, rows


def hash_quality(p: int = 12, trials: int = 100):
    """The error band on sequential integers and on a cent grid by hash."""
    rows = []
    m = 1 << p
    for kind in ("sequential", "grid"):
        for name, f in (("splitmix", H.splitmix), ("FixedHasher", H.fixed_hasher_u64),
                        ("identity", H.identity)):
            e = []
            for t in range(trials):
                v = values(kind, 20 * m, t)
                s = H.HLL(p, hashf=f)
                s.update(v)
                e.append(s.count() / truth(v) - 1)
            e = np.array(e)
            rows.append((kind, name, p, trials, float(np.std(e)), float(np.mean(e)),
                         1.04 / np.sqrt(m)))
    return ("values", "hash", "p", "trials", "sd", "bias", "1.04/sqrt(m)"), rows


def merge_checks():
    """Merge is the register-wise maximum and equals the sketch of the union;
    and order and chunking cannot move a register (rule 3)."""
    rows = []
    for p in (4, 8, 12, 14):
        a = values("random", 30_000, 7)
        b = np.concatenate([a[:10_000], values("random", 25_000, 8)])  # overlapping
        sa, sb, su = H.HLL(p), H.HLL(p), H.HLL(p)
        sa.update(a)
        sb.update(b)
        su.update(np.concatenate([a, b]))
        mg = sa.merge(sb)
        exact_max = bool(np.array_equal(mg.reg, np.maximum(sa.reg, sb.reg)))
        union = bool(np.array_equal(mg.reg, su.reg)) and mg.count() == su.count()
        comm = bool(np.array_equal(sa.merge(sb).reg, sb.merge(sa).reg))
        idem = bool(np.array_equal(sa.merge(sa).reg, sa.reg))
        ds_a = HyperLogLogPlusPlus(p=p, hashfunc=H.splitmix_int, reg=sa.reg.copy())
        ds_b = HyperLogLogPlusPlus(p=p, hashfunc=H.splitmix_int, reg=sb.reg.copy())
        ds_a.merge(ds_b)
        ds_same = bool(np.array_equal(ds_a.reg, mg.reg))
        allv = np.concatenate([a, b])
        perm = np.random.default_rng(1).permutation(len(allv))
        chunked = H.HLL(p)
        for piece in np.array_split(allv[perm], 977):
            chunked.update(piece)
        order = bool(np.array_equal(chunked.reg, su.reg))
        rows.append((p, exact_max, union, comm, idem, ds_same, order,
                     abs(mg.count() / truth(allv) - 1)))
    hdr = ("p", "merge = max", "merge = sketch of union", "commutative", "idempotent",
           "= ds merge", "shuffled, 977 chunks = one", "union rel err")
    return hdr, rows


def keying():
    """-0.0 and 0.0 are one value; NaN payloads (and the null) are kept apart,
    as audit's exact counter keeps them."""
    nan2 = np.array([0x7FF8_0000_0000_0001], np.uint64).view(np.float64)[0]
    null = np.array([H.NULL_BITS], np.uint64).view(np.float64)[0]
    v = np.array([0.0, -0.0, np.nan, nan2, null, 1.0, 1.0])
    s = H.HLL(12)
    s.update(v)
    return [("values", "0.0, -0.0, NaN, NaN (other payload), null, 1.0, 1.0"),
            ("distinct keys", truth(v)), ("estimate", round(s.count(), 6))]


def zero_weight():
    """Rule 9 against audit's own convention: audit counts a weight-0 row's
    value (its docstring); a sketch that skipped them would count a different
    thing from the exact counter it continues past the cap."""
    v = values("random", 5_000, 9)
    w = np.ones(5_000)
    w[::3] = 0.0
    a, b = H.HLL(12), H.HLL(12)
    a.update(v, w, count_zero_weight=True)
    b.update(v, w, count_zero_weight=False)
    return [("rows", 5_000), ("weight 0", int((w == 0).sum())),
            ("distinct, every row (audit's exact counter)", truth(v)),
            ("estimate counting weight-0 rows", round(a.count(), 1)),
            ("distinct, positive weight only", truth(v[w > 0])),
            ("estimate skipping weight-0 rows", round(b.count(), 1))]


def determinism():
    """The same registers in fresh processes under different PYTHONHASHSEEDs
    (Python's own hash is salted per process; this sketch uses none of it),
    and pinned hash values."""
    here = os.path.dirname(os.path.abspath(H.__file__))
    code = (f"import sys; sys.path.insert(0, {here!r}); import hll, numpy as np, hashlib;"
            "s = hll.HLL(12); s.update(np.random.default_rng(0).normal(size=50000));"
            "print(hashlib.sha256(s.reg.tobytes()).hexdigest()[:16])")
    digests = set()
    for seed in ("0", "1", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        digests.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                   text=True, check=True).stdout.strip())
    pins = [
        ("splitmix(0) = SplitMix64's first output from state 0", hex(H.splitmix_int(0)),
         H.splitmix_int(0) == 0xE220A8397B1DCDAF),
        ("FixedHasher(7u64) = hasher.rs's pinned 0x5201ec7d2d0b0310",
         hex(int(H.fixed_hasher_u64(np.array([7], np.uint64))[0])),
         int(H.fixed_hasher_u64(np.array([7], np.uint64))[0]) == 0x5201EC7D2D0B0310),
    ]
    return digests, pins


def sizes():
    rows = []
    for p in (4, 6, 8, 10, 12, 14, 16):
        m = 1 << p
        rows.append((p, m, m, m * 6 // 8, HyperLogLog(p=p).bytesize(), 1.04 / np.sqrt(m)))
    hdr = ("p", "registers", "bytes (u8 each)", "bytes (6-bit packed)",
           "datasketch bytesize()", "1.04/sqrt(m)")
    return hdr, rows
