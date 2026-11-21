#!/usr/bin/env python
"""Study 544, part 2: expected shortfall from a DDSketch-style histogram, and t-digest
rank error, against exact numpy values and the installed oracles.  Python only;
every stream is seeded (reuses ``study_544.stream``) and no data file is written.

    python scripts/study_544_tails.py es      [--seeds 20]   # ES, VaR vs exact, N = 10k, 100k
    python scripts/study_544_tails.py decay   [--seeds 20]   # lambda = 0.999, exact weighted tail mean
    python scripts/study_544_tails.py window  [--seeds 20]   # last W = 5000 rows, exact removal
    python scripts/study_544_tails.py buckets [--seeds 20]   # live buckets, state bytes, ddsketch oracle
    python scripts/study_544_tails.py tdigest [--seeds 20]   # rank error, compression 100 and 200

Run in ~/study544-venv (see study_544.py).

DDSketch rule.  alpha = tanh(1/128), gamma = (1 + alpha)/(1 - alpha).  A value x > 0
goes to bucket i = ceil(log(x)/log(gamma)), which covers (gamma^(i-1), gamma^i]; the
bucket's representative is r_i = 2 gamma^i/(gamma + 1).  Then r_i/gamma^i = 1 - alpha
and r_i/gamma^(i-1) = 1 + alpha, so |r_i - x| <= alpha x for every x in the bucket.
Negative values use a second store on |x| (representative -r_i); zero has a count.

Tail mean at mass m (m = q N rows, or q W for weights): walk the buckets in value order,
take whole buckets until the next one would pass m, and take the VaR bucket's partial
mass (m minus the mass already taken) at its representative.  Pairing the k smallest
values with the k slots in sorted order puts each pair in one bucket, so
|ES_hat - ES| <= alpha * mean(|x|) over the tail (= alpha |ES| when the tail has one sign).
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from study_544 import DISTRIBUTIONS, stream

ALPHA = math.tanh(1.0 / 128.0)
GAMMA = (1.0 + ALPHA) / (1.0 - ALPHA)
LOG_GAMMA = math.log(GAMMA)
HEADER_BYTES = 40  # alpha, zero count, total weight, two store lengths: 5 f64 (a stated choice)
BUCKET_BYTES = 16  # key (i64) + count (f64)
OFFSET = 1 << 20
# (name, side, q): the tail holds the fraction q of the mass; lower = smallest, upper = largest
LEVELS = (("0.01 lower", "lower", 0.01), ("0.05 lower", "lower", 0.05),
          ("0.95 upper", "upper", 0.05), ("0.99 upper", "upper", 0.01))


def keys_of(x: np.ndarray) -> np.ndarray:
    """Signed bucket key, ascending in value: negatives -(OFFSET+i), zero 0, positives OFFSET+i."""
    ax = np.abs(x)
    nz = ax > 0
    i = np.zeros(x.shape, dtype=np.int64)
    i[nz] = np.ceil(np.log(ax[nz]) / LOG_GAMMA).astype(np.int64)
    k = np.zeros(x.shape, dtype=np.int64)
    k[nz] = np.where(x[nz] > 0, OFFSET + i[nz], -(OFFSET + i[nz]))
    return k


def rep_of(keys: np.ndarray) -> np.ndarray:
    i = np.abs(keys) - OFFSET
    r = 2.0 * np.power(GAMMA, i.astype(np.float64)) / (GAMMA + 1.0)
    return np.where(keys == 0, 0.0, np.where(keys > 0, r, -r))


def sketch_of(x: np.ndarray, w: np.ndarray | None = None):
    """(sorted unique keys, mass) of the stored buckets."""
    k = keys_of(x)
    u, inv = np.unique(k, return_inverse=True)
    return u, np.bincount(inv, weights=w, minlength=len(u)).astype(np.float64)


def tail_from(rep: np.ndarray, mass: np.ndarray, m: float):
    """(ES with partial VaR bucket, VaR, ES with the whole VaR bucket, mean|x| of the tail)."""
    cum = np.cumsum(mass)
    j = int(np.searchsorted(cum, m * (1.0 - 1e-15), side="left"))
    j = min(j, len(rep) - 1)
    before = cum[j] - mass[j]
    part = m - before
    es = (np.dot(mass[:j], rep[:j]) + part * rep[j]) / m
    mabs = (np.dot(mass[:j], np.abs(rep[:j])) + part * abs(rep[j])) / m
    full = np.dot(mass[: j + 1], rep[: j + 1]) / cum[j]
    return es, rep[j], full, mabs


def exact_tail(x: np.ndarray, w: np.ndarray | None, m: float):
    """Exact (ES, VaR, mean|x|) with the same fractional mass at the VaR value."""
    o = np.argsort(x, kind="stable")
    xs = x[o]
    ws = np.ones_like(xs) if w is None else w[o]
    cum = np.cumsum(ws)
    j = min(int(np.searchsorted(cum, m * (1.0 - 1e-15), side="left")), len(xs) - 1)
    part = m - (cum[j] - ws[j])
    es = (np.dot(ws[:j], xs[:j]) + part * xs[j]) / m
    mabs = (np.dot(ws[:j], np.abs(xs[:j])) + part * abs(xs[j])) / m
    return es, xs[j], mabs


def level_pair(x, w, side, q):
    """(sketch result, exact result) for one level; upper tail = lower tail of -x (mirror)."""
    y = x if side == "lower" else -x
    total = float(len(y)) if w is None else float(w.sum())
    m = q * total
    keys, mass = sketch_of(y, w)
    es_s, var_s, full_s, mabs_s = tail_from(rep_of(keys), mass, m)
    es_e, var_e, mabs_e = exact_tail(y, w, m)
    sg = 1.0 if side == "lower" else -1.0
    return (sg * es_s, sg * var_s, sg * full_s), (sg * es_e, sg * var_e, mabs_e)


class Agg:
    """Largest relative gap of the ES to the exact ES, to the bound, of the VaR, of the full-bucket ES."""

    def __init__(self) -> None:
        self.gap = self.ratio = self.var = self.full = 0.0
        self.viol = 0
        self.n = 0

    def add(self, s, e):
        gap = abs(s[0] - e[0])
        self.gap = max(self.gap, gap / abs(e[0]))
        r = gap / (ALPHA * e[2])
        self.ratio = max(self.ratio, r)
        self.viol += r > 1.0 + 1e-9
        self.var = max(self.var, abs(s[1] - e[1]) / abs(e[1]))
        self.full = max(self.full, abs(s[2] - e[0]) / abs(e[0]))
        self.n += 1


HDR = (f"{'dist':<10} {'level':<11} {'runs':>5} {'max gap/ES':>11} {'gap/bound':>10} "
       f"{'viol':>5} {'max VaR rel':>12} {'whole-bucket gap':>17}")


def row(dist, level, a: Agg):
    return (f"{dist:<10} {level:<11} {a.n:>5} {a.gap:>11.3e} {a.ratio:>10.3f} {a.viol:>5} "
            f"{a.var:>12.3e} {a.full:>17.3e}")


def run_es(seeds: int) -> None:
    print(f"alpha={ALPHA!r} gamma={GAMMA!r} bound alpha*mean|x| (= alpha*|ES| for a one-signed tail)")
    for n in (10_000, 100_000):
        print(f"\nN = {n}, seeds = {seeds}")
        print(HDR)
        worst = 0.0
        for dist in DISTRIBUTIONS:
            for name, side, q in LEVELS:
                a = Agg()
                for seed in range(seeds):
                    x = stream(dist, seed, n)
                    a.add(*level_pair(x, None, side, q))
                print(row(dist, name, a))
                worst = max(worst, a.gap)
        print(f"largest gap/ES over all: {worst:.3e}   alpha = {ALPHA:.3e}")


def run_decay(seeds: int, lam: float = 0.999) -> None:
    print(f"lambda = {lam}; weights lam^(N-1-t) on the row clock; exact = weighted tail mean, "
          f"same fractional mass q*sum(w)")
    for n in (10_000, 100_000):
        print(f"\nN = {n}, seeds = {seeds}, n_eff = {1 / (1 - lam):.0f}")
        print(HDR)
        w = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        for dist in DISTRIBUTIONS:
            for name, side, q in LEVELS:
                a = Agg()
                for seed in range(seeds):
                    a.add(*level_pair(stream(dist, seed, n), w, side, q))
                print(row(dist, name, a))
    # the batch weights equal the streaming recursion c <- lam*c + onehot(bucket), up to rounding
    print("\nstreaming recursion (every bucket times lambda per row, then +1) vs the batch weights,")
    print("N = 10000, seed 0: largest |difference| / the bucket's weight")
    n = 10_000
    w = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
    for dist in DISTRIBUTIONS:
        x = stream(dist, 0, n)
        u, inv = np.unique(keys_of(x), return_inverse=True)
        c = np.zeros(len(u))
        for t in range(n):
            c *= lam
            c[inv[t]] += 1.0
        b = np.bincount(inv, weights=w, minlength=len(u))
        print(f"  {dist:<10} buckets={len(u):>4}  max rel diff = {np.max(np.abs(c - b) / b):.2e}")


def run_window(seeds: int, n: int = 100_000, wsize: int = 5000, step: int = 5000) -> None:
    print(f"window W = {wsize}, N = {n}, query every {step} rows from the first full window; "
          f"counts removed exactly (integer subtract); seeds = {seeds}")
    print(HDR + f" {'max live':>9}")
    for dist in DISTRIBUTIONS:
        aggs = {nm: Agg() for nm, _, _ in LEVELS}
        live_max = 0
        exact_counts = 0
        for seed in range(seeds):
            x = stream(dist, seed, n)
            ky = {"lower": keys_of(x), "upper": keys_of(-x)}
            tabs = {}
            for side in ("lower", "upper"):
                u, inv = np.unique(ky[side], return_inverse=True)
                tabs[side] = (u, inv.tolist(), [0] * len(u), rep_of(u))
            for t in range(n):
                for side in ("lower", "upper"):
                    _, inv, cnt, _ = tabs[side]
                    cnt[inv[t]] += 1
                    if t >= wsize:
                        cnt[inv[t - wsize]] -= 1
                if t >= wsize - 1 and (t - (wsize - 1)) % step == 0:
                    win = x[t - wsize + 1 : t + 1]
                    for nm, side, q in LEVELS:
                        u, inv, cnt, rp = tabs[side]
                        c = np.array(cnt, dtype=np.float64)
                        y = win if side == "lower" else -win
                        ref = np.bincount(inv_of(u, keys_of(y)), minlength=len(u))
                        exact_counts += bool(np.array_equal(ref, c))
                        live = c > 0
                        live_max = max(live_max, int(live.sum()))
                        m = q * wsize
                        es_s, var_s, full_s, _ = tail_from(rp[live], c[live], m)
                        es_e, var_e, mabs = exact_tail(y, None, m)
                        sg = 1.0 if side == "lower" else -1.0
                        aggs[nm].add((sg * es_s, sg * var_s, sg * full_s), (sg * es_e, sg * var_e, mabs))
        for nm, _, _ in LEVELS:
            print(row(dist, nm, aggs[nm]) + f" {live_max:>9}")
        print(f"  {dist}: windowed counts equal the bincount of the window at "
              f"{exact_counts} of {seeds * 20 * len(LEVELS)} checks")


def inv_of(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.searchsorted(u, k)


def run_buckets(seeds: int) -> None:
    from ddsketch.ddsketch import BaseDDSketch
    from ddsketch.mapping import LogarithmicMapping
    from ddsketch.store import DenseStore

    print("live buckets (positive + negative + zero) of one signed sketch; bytes = live*16 + "
          f"{HEADER_BYTES} header")
    for n in (10_000, 100_000, 1_000_000):
        print(f"\nN = {n}")
        print(f"{'dist':<10} {'min':>5} {'mean':>7} {'max':>5} {'max bytes':>10} {'tail-0.99 buckets (mean)':>26}")
        for dist in DISTRIBUTIONS:
            nb, tb = [], []
            for seed in range(seeds if n < 1_000_000 else 2):
                x = stream(dist, seed, n)
                u, c = sketch_of(x)
                nb.append(len(u))
                # buckets that hold the top 1% of the mass
                cum = np.cumsum(c[::-1])
                tb.append(int(np.searchsorted(cum, 0.01 * n)) + 1)
            print(f"{dist:<10} {min(nb):>5} {np.mean(nb):>7.1f} {max(nb):>5} "
                  f"{max(nb) * BUCKET_BYTES + HEADER_BYTES:>10} {np.mean(tb):>26.1f}")
    print("\nddsketch 3.0.1 oracle (LogarithmicMapping(alpha), unbounded DenseStore), N = 100000:")
    print("  bucket counts compared with this module's, key by key; VaR within alpha of exact")
    mism = tot = 0
    worst = 0.0
    for dist in DISTRIBUTIONS:
        for seed in range(min(seeds, 5)):
            x = stream(dist, seed, 100_000)
            s = BaseDDSketch(LogarithmicMapping(ALPHA), DenseStore(), DenseStore(), 0.0)
            for v in x.tolist():
                s.add(v)
            mine = {}
            u, c = sketch_of(x)
            for k, cc in zip(u.tolist(), c.tolist()):
                mine[(1 if k > 0 else -1, abs(k) - OFFSET)] = int(cc)
            theirs = oracle_bins(s)
            tot += len(mine)
            mism += sum(1 for k in set(mine) | set(theirs) if mine.get(k, 0) != theirs.get(k, 0))
            for p in (0.01, 0.05, 0.5, 0.95, 0.99):
                ex = float(np.quantile(x, p, method="inverted_cdf"))
                worst = max(worst, abs(s.get_quantile_value(p) - ex) / abs(ex))
    print(f"  buckets compared {tot}, mismatching {mism}; largest ddsketch quantile rel. error "
          f"vs inverted_cdf {worst:.3e} (alpha {ALPHA:.3e})")


def oracle_bins(s) -> dict:
    out = {}
    for sg, st in ((1, s._store), (-1, s._negative_store)):
        for idx, cnt in enumerate(st.bins):
            if cnt > 0:
                out[(sg, idx + st.offset)] = int(cnt)
    return out


# ------------------------------------------------------------------------------ t-digest


class MergingDigest:
    """Dunning-Ertl merging t-digest, scale function k1 (asin), numpy/Python floats.

    k(q) = delta/(2 pi) asin(2q - 1); a centroid may absorb neighbours while the k-span of its
    mass stays <= 1.  A buffer of 8 delta points is merged into the centroids when full.
    State: centroids (mean, weight) = 16 bytes each, plus min and max.
    """

    def __init__(self, delta: float) -> None:
        self.delta = float(delta)
        self.means = np.empty(0)
        self.weights = np.empty(0)
        self.buf: list[float] = []
        self.cap = int(8 * delta)
        self.lo, self.hi = math.inf, -math.inf
        self.total = 0.0

    def _k(self, q: float) -> float:
        return self.delta / (2.0 * math.pi) * math.asin(min(1.0, max(-1.0, 2.0 * q - 1.0)))

    def _q(self, k: float) -> float:
        if k >= self.delta / 4.0:
            return 1.0
        return (math.sin(2.0 * math.pi * k / self.delta) + 1.0) / 2.0

    def update_many(self, xs: np.ndarray) -> None:
        for i in range(0, len(xs), self.cap):
            self.buf = xs[i : i + self.cap].tolist()
            self._merge()

    def _merge(self) -> None:
        if not self.buf:
            return
        b = np.sort(np.asarray(self.buf))
        self.lo, self.hi = min(self.lo, float(b[0])), max(self.hi, float(b[-1]))
        m = np.concatenate([self.means, b])
        w = np.concatenate([self.weights, np.ones(len(b))])
        o = np.argsort(m, kind="stable")
        m, w = m[o].tolist(), w[o].tolist()
        total = sum(w)
        self.total = total
        om, ow = [], []
        cum = 0.0
        qlim = self._q(self._k(0.0) + 1.0)
        cm, cw = m[0], w[0]
        for mi, wi in zip(m[1:], w[1:]):
            if (cum + cw + wi) / total <= qlim:
                cw += wi
                cm += (mi - cm) * wi / cw
            else:
                om.append(cm)
                ow.append(cw)
                cum += cw
                qlim = self._q(self._k(cum / total) + 1.0)
                cm, cw = mi, wi
        om.append(cm)
        ow.append(cw)
        self.means, self.weights = np.array(om), np.array(ow)
        self.buf = []

    def quantile(self, p: float) -> float:
        m, w, total = self.means, self.weights, self.total
        t = p * total
        if len(m) == 1:
            return float(m[0])
        if t < w[0] / 2.0:
            return float(self.lo + (m[0] - self.lo) * t / (w[0] / 2.0)) if w[0] > 1 else float(m[0])
        if t > total - w[-1] / 2.0:
            r = total - t
            return float(self.hi - (self.hi - m[-1]) * r / (w[-1] / 2.0)) if w[-1] > 1 else float(m[-1])
        centers = np.cumsum(w) - w / 2.0
        i = int(np.searchsorted(centers, t, side="right")) - 1
        f = (t - centers[i]) / (centers[i + 1] - centers[i])
        return float(m[i] + f * (m[i + 1] - m[i]))

    @property
    def state_bytes(self) -> int:
        return 16 * len(self.means)


PS = (0.01, 0.05, 0.5, 0.95, 0.99)
TD_SEEDS = 3  # seeds run through the pure-Python tdigest oracle


def rank_err(sorted_x: np.ndarray, q: float, p: float) -> float:
    return abs(np.searchsorted(sorted_x, q, side="right") / len(sorted_x) - p)


def run_tdigest(seeds: int, n: int = 100_000) -> None:
    import crick
    import pytdigest
    import tdigest

    impls = ("mine", "pytdigest", "crick", "tdigest")
    print(f"rank error |F(q_hat) - p| with F the empirical CDF of the stream; N = {n}, seeds = {seeds}")
    print("  mine = Dunning-Ertl merging digest, k1, buffer 8 delta (this module); pytdigest 0.1.4; "
          "crick 0.0.8; tdigest 0.5.2.2 with delta = 1/compression, K = 25 (an older, tree-based algorithm; first "
          f"{TD_SEEDS} seeds only)")
    res = {(c, im): {"err": {p: 0.0 for p in PS}, "cent": [], "dist": {}} for c in (100, 200) for im in impls}
    agree = {c: {"rank": 0.0, "val": 0.0} for c in (100, 200)}  # pytdigest vs crick
    agree_mine = {c: 0.0 for c in (100, 200)}
    for dist in DISTRIBUTIONS:
        for c in (100, 200):
            for im in impls:
                res[(c, im)]["dist"][dist] = {p: 0.0 for p in PS}
        for seed in range(seeds):
            x = stream(dist, seed, n)
            sx = np.sort(x)
            for c in (100, 200):
                md = MergingDigest(c)
                md.update_many(x)
                pt = pytdigest.TDigest.compute(x, compression=c)
                cr = crick.TDigest(compression=c)
                cr.update(x)
                td = tdigest.TDigest(delta=1.0 / c)
                if seed < TD_SEEDS:  # the pure-Python oracle takes 3 to 6 s a stream
                    td.batch_update(x)
                qs = {
                    "mine": [md.quantile(p) for p in PS],
                    "pytdigest": [pt.inverse_cdf(p) for p in PS],
                    "crick": [float(cr.quantile(p)) for p in PS],
                    "tdigest": [td.percentile(100 * p) if seed < TD_SEEDS else 0.0 for p in PS],
                }
                ncent = {"mine": len(md.means), "pytdigest": len(pt.get_centroids()),
                         "crick": len(cr.centroids()), "tdigest": len(td)}
                for im in impls:
                    if im == "tdigest" and seed >= TD_SEEDS:
                        continue
                    r = res[(c, im)]
                    r["cent"].append(ncent[im])
                    for p, q in zip(PS, qs[im]):
                        e = rank_err(sx, q, p)
                        r["err"][p] = max(r["err"][p], e)
                        r["dist"][dist][p] = max(r["dist"][dist][p], e)
                for p, a, b, m_ in zip(PS, qs["pytdigest"], qs["crick"], qs["mine"]):
                    agree[c]["rank"] = max(agree[c]["rank"], abs(np.searchsorted(sx, a, "right")
                                           - np.searchsorted(sx, b, "right")) / n)
                    agree[c]["val"] = max(agree[c]["val"], abs(a - b) / max(abs(a), 1e-12))
                    agree_mine[c] = max(agree_mine[c], abs(np.searchsorted(sx, a, "right")
                                        - np.searchsorted(sx, m_, "right")) / n)
    for c in (100, 200):
        print(f"\ncompression {c}: largest rank error over 4 distributions x {seeds} seeds, by p")
        print(f"{'impl':<10} " + " ".join(f"p={p:<6}" for p in PS) + f" {'max':>9} {'centroids min/mean/max':>24} {'state bytes (max)':>18}")
        for im in impls:
            r = res[(c, im)]
            mx = max(r["err"].values())
            ce = r["cent"]
            print(f"{im:<10} " + " ".join(f"{r['err'][p]:<8.2e}" for p in PS)
                  + f" {mx:>9.2e} {min(ce):>7}/{np.mean(ce):>7.1f}/{max(ce):>7} {16 * max(ce):>18}")
        print(f"  pytdigest vs crick: largest quantile rank difference {agree[c]['rank']:.2e}, "
              f"largest relative value difference {agree[c]['val']:.2e}")
        print(f"  pytdigest vs mine: largest rank difference {agree_mine[c]:.2e}")
        print("  per distribution (largest over p) :", {d: {im: f"{max(res[(c, im)]['dist'][d].values()):.1e}" for im in impls} for d in DISTRIBUTIONS})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["es", "decay", "window", "buckets", "tdigest"])
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    {"es": run_es, "decay": run_decay, "window": run_window, "buckets": run_buckets,
     "tdigest": run_tdigest}[a.what](a.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
