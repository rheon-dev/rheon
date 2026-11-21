#!/usr/bin/env python
"""Study 544: do rheon's streaming tail measures agree with independent oracles?

One entry point.  Python only; nothing here touches the Rust core.  Every generator is
seeded and no data file is written.

    python scripts/study_544.py psquared [--n 1000] [--seeds 50] [--trajectory]
        P-squared (Jain and Chlamtac 1985), a numpy port against river's ``stats.Quantile``.
    python scripts/study_544.py tails    [--seeds 20] [--only es|decay|window|buckets]
        expected shortfall (the tail mean) and VaR from a DDSketch-style histogram against
        exact numpy values: no decay, decay (lambda = 0.999), a window (W = 5000), the
        live buckets, and the ddsketch package as an oracle for the bucket rule.
    python scripts/study_544.py tdigest  [--seeds 20] [--td-seeds 20] [--workers 4]
        t-digest rank error against exact ranks for a merging digest written from the
        paper, pytdigest, crick and tdigest; compression 100 and 200.
    python scripts/study_544.py all [--quick]        # the three above
    python scripts/study_544.py oracles              # package, version, licence, URL
    python scripts/study_544.py state                # bytes of each online form's state

``--quick`` shrinks every part (a few seeds, smaller N, tdigest on one seed) and finishes
in well under two minutes.  Each part checks its result against the tolerances in the
TOLERANCES block below and the run exits 1 on any failure; the full run's numbers and the
reasons for each tolerance are in docs/records/STUDY-544-TAILS.md.

Run it in a venv holding the oracles (this study used ~/study544-venv, not the project's
.venv, so the project environment stays untouched):

    uv pip install river pytdigest tdigest ddsketch crick datasketches numpy

DDSketch rule.  alpha = tanh(1/128), gamma = (1 + alpha)/(1 - alpha).  A value x > 0
goes to bucket i = ceil(log(x)/log(gamma)), which covers (gamma^(i-1), gamma^i]; the
bucket's representative is r_i = 2 gamma^i/(gamma + 1).  Then r_i/gamma^i = 1 - alpha
and r_i/gamma^(i-1) = 1 + alpha, so |r_i - x| <= alpha x for every x in the bucket.
Negative values use a second store on |x| (representative -r_i); zero has a count.

Tail mean at mass m (m = q N rows, or q * sum(w) for weights): walk the buckets in value
order, take whole buckets until the next one would pass m, and take the VaR bucket's
partial mass (m minus the mass already taken) at its representative.  Pairing the k
smallest values with the k slots in sorted order puts each pair in one bucket, so
|ES_hat - ES| <= alpha * mean(|x|) over the tail (= alpha |ES| when the tail has one sign).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from importlib.metadata import PackageNotFoundError, metadata, version
from multiprocessing import Pool

import numpy as np

ORACLES = ["river", "pytdigest", "tdigest", "ddsketch", "crick", "datasketches", "numpy", "scipy"]
QUANTILES = (0.05, 0.5, 0.95)
DISTRIBUTIONS = ("normal", "lognormal", "student_t3", "pareto2.5")

ALPHA = math.tanh(1.0 / 128.0)
GAMMA = (1.0 + ALPHA) / (1.0 - ALPHA)
LOG_GAMMA = math.log(GAMMA)
HEADER_BYTES = 40  # alpha, zero count, total weight, two store lengths: 5 f64 (a stated assumption)
BUCKET_BYTES = 16  # key (i64) + count (f64)
OFFSET = 1 << 20
# (name, side, q): the tail holds the fraction q of the mass; lower = smallest, upper = largest
LEVELS = (
    ("0.01 lower", "lower", 0.01),
    ("0.05 lower", "lower", 0.05),
    ("0.95 upper", "upper", 0.05),
    ("0.99 upper", "upper", 0.01),
)

# ------------------------------------------------------------------------- TOLERANCES
# (a) P-squared against river: bit equality, in ulps.
P2_TOL_ULPS = 0
# (b) ES from the sketch against the exact tail mean, relative.  The deterministic bound is
#     |ES_hat - ES| <= alpha * mean|x| over the tail (alpha |ES| for a one-signed tail).  The
#     tolerance is a fraction of that bound: half of it for the light tails (measured at most
#     0.18 with no decay, 0.34 with decay, 0.38 in a window), all of it for Pareto, whose lower
#     0.01 tail sits on a bucket edge and measures 0.76 to 0.88 of the bound.
ES_TOL_FRACTION = {"normal": 0.5, "lognormal": 0.5, "student_t3": 0.5, "pareto2.5": 1.0}
# (c) windowed counts against the bincount of the window: exact (integer subtraction).
# (d) t-digest rank error |F(q_hat) - p| against exact ranks: the measured maximum of the
#     merging digest written from the paper (20 seeds x 4 distributions, N = 100,000), rounded
#     up to two significant figures, times 1.5.
TD_MEASURED_MAX = {100: 2.5e-3, 200: 1.2e-3}
TD_MARGIN = 1.5
TD_TOL = {c: TD_MARGIN * v for c, v in TD_MEASURED_MAX.items()}


# --------------------------------------------------------------------------- oracles


def oracle_table() -> list[dict[str, str]]:
    rows = []
    for name in ORACLES:
        try:
            m = metadata(name)
        except PackageNotFoundError:
            rows.append({"package": name, "version": "NOT INSTALLED"})
            continue
        lic = m.get("License-Expression") or ""
        if not lic:
            lic = "; ".join(
                c.split("::")[-1].strip() for c in (m.get_all("Classifier") or []) if "License" in c
            )
        if not lic:
            lic = (m.get("License") or "").strip().splitlines()[0][:40] if m.get("License") else "?"
        urls = m.get("Home-page") or "; ".join(m.get_all("Project-URL") or [])[:100]
        rows.append({"package": name, "version": version(name), "licence": lic, "url": urls})
    return rows


# ------------------------------------------------------------------ P-squared (numpy)


class P2:
    """Jain and Chlamtac's P-squared estimator of one quantile, in numpy float64.

    State: five marker heights q, five integer marker positions n, five desired
    positions n', and the five increments dn (fixed by p).  The arithmetic order
    of the parabolic formula matches the one river uses, so the two agree to the
    bit wherever river follows the paper.
    """

    def __init__(self, p: float) -> None:
        if not 0 < p < 1:
            raise ValueError("p must lie in (0, 1)")
        self.p = p
        self.dn = np.array([0.0, p / 2, p, (1 + p) / 2, 1.0])  # increments
        self.nd = np.array([1.0, 1 + 2 * p, 1 + 4 * p, 3 + 2 * p, 5.0])  # desired positions
        self.n = np.arange(1, 6, dtype=np.int64)  # actual positions
        self.q: list[float] = []  # heights; first five observations, then sorted
        self.sorted = False

    def _cell(self, x: float) -> int:
        q = self.q
        if x < q[0]:
            q[0] = x
            return 1
        for i in range(1, 5):
            if q[i - 1] <= x < q[i]:
                return i
        if q[4] < x:
            q[4] = x
        return 4

    def update(self, x: float) -> None:
        x = np.float64(x)
        if len(self.q) != 5:
            self.q.append(x)
            return
        if not self.sorted:
            self.q.sort()
            self.sorted = True
        k = self._cell(x)
        self.n[k:] += 1  # positions of markers k+1..5 (1-based) move up
        self.nd += self.dn
        q, n = self.q, self.n
        for i in (1, 2, 3):
            d = self.nd[i] - n[i]
            if (d >= 1 and n[i + 1] - n[i] > 1) or (d <= -1 and n[i - 1] - n[i] < -1):
                s = 1 if d >= 0 else -1
                ni, nm, npl = np.float64(n[i]), np.float64(n[i - 1]), np.float64(n[i + 1])
                # parabolic (P^2) prediction
                outer = s / (npl - nm)
                left = (ni - nm + s) * (q[i + 1] - q[i]) / (npl - ni)
                right = (npl - ni - s) * (q[i] - q[i - 1]) / (ni - nm)
                qn = q[i] + outer * (left + right)
                if q[i - 1] < qn < q[i + 1]:
                    q[i] = qn
                else:  # linear fallback
                    q[i] = q[i] + s * (q[i + s] - q[i]) / (np.float64(n[i + s]) - ni)
                n[i] += s

    def get(self) -> float | None:
        if self.sorted:
            return self.q[2]
        if self.q:
            self.q.sort()
            m = len(self.q)
            return self.q[int(min(max(m - 1, 0), m * self.p))]
        return None


def state_bytes() -> dict[str, int]:
    f64 = 8
    return {
        "markers_q": 5 * f64,
        "positions_n": 5 * f64,  # integers in the paper; stored as f64 or i64, 8 bytes either way
        "desired_n_prime": 5 * f64,
        "increments_dn": 5 * f64,
        "p": f64,
        "count": f64,
        "total_full": 5 * 4 * f64 + 2 * f64,
        "total_minimal_q_n_p_count": 2 * 5 * f64 + 2 * f64,  # dn and n' re-derivable from p, count
    }


def print_state() -> None:
    print("P-squared, bytes per quantile")
    for k, v in state_bytes().items():
        print(f"  {k:<28}{v}")
    print(
        f"DDSketch histogram, bytes = live buckets * {BUCKET_BYTES} + {HEADER_BYTES} "
        "(header: an assumption)"
    )
    for b in (227, 1292):
        print(f"  {b:>5} buckets -> {b * BUCKET_BYTES + HEADER_BYTES:>6} bytes")
    print("t-digest, bytes = centroids * 16 (mean, weight) [+ 24 for min, max, total if stored]")
    for c, lo, hi in ((100, 61, 64), (200, 107, 118)):
        print(f"  compression {c}: {lo} to {hi} centroids -> {16 * lo} to {16 * hi} bytes")


# ------------------------------------------------------------------------ comparison


def ulps(a: float, b: float) -> int:
    """Distance in representable doubles between a and b (sign-magnitude order)."""

    def key(x: float) -> int:
        i = int(np.float64(x).view(np.int64))
        return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)

    return abs(key(a) - key(b))


def stream(dist: str, seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng([seed, DISTRIBUTIONS.index(dist), 544])
    if dist == "normal":
        return rng.standard_normal(n)
    if dist == "lognormal":
        return rng.lognormal(0.0, 1.0, n)
    if dist == "student_t3":
        return rng.standard_t(3, n)
    if dist == "pareto2.5":
        return rng.pareto(2.5, n) + 1.0  # classical Pareto, x_m = 1
    raise ValueError(dist)


def run_psquared(n: int, seeds: int, trajectory: bool) -> list[str]:
    from river import stats

    print(
        f"river {version('river')}  numpy {np.__version__}  n={n}  seeds={seeds}  "
        f"trajectory={trajectory}"
    )
    total_cmp = 0
    total_mis = 0
    runs_mis = 0
    worst = 0
    worst_where = None
    rows = []
    for dist in DISTRIBUTIONS:
        for p in QUANTILES:
            cmp_n = mis = run_mis = dmax = 0
            maxabs = 0.0
            for seed in range(seeds):
                xs = stream(dist, seed, n).tolist()
                r, m = stats.Quantile(p), P2(p)
                bad = False
                for t, x in enumerate(xs):
                    r.update(x)
                    m.update(x)
                    if trajectory or t == n - 1:
                        a, b = r.get(), m.get()
                        cmp_n += 1
                        if a != b:
                            u = ulps(a, b)
                            mis += 1
                            bad = True
                            maxabs = max(maxabs, abs(a - b))
                            if u > dmax:
                                dmax = u
                            if u > worst:
                                worst, worst_where = u, (dist, p, seed, t)
                run_mis += bad
            rows.append((dist, p, cmp_n, mis, run_mis, dmax, maxabs))
            total_cmp += cmp_n
            total_mis += mis
            runs_mis += run_mis
    print(
        f"{'dist':<10} {'p':>5} {'compared':>9} {'mismatch':>9} {'runs':>5} "
        f"{'max ulps':>9} {'max abs':>10}"
    )
    for dist, p, c, mi, rm, d, ab in rows:
        print(f"{dist:<10} {p:>5} {c:>9} {mi:>9} {rm:>5} {d:>9} {ab:>10.3g}")
    print(
        f"TOTAL compared={total_cmp} mismatches={total_mis} runs_with_mismatch={runs_mis}/"
        f"{len(rows) * seeds} largest_ulps={worst} at {worst_where}"
    )
    if total_mis or worst > P2_TOL_ULPS:
        return [f"psquared: {total_mis} mismatches, largest {worst} ulps (tolerance {P2_TOL_ULPS})"]
    return []


# ----------------------------------------------------------- DDSketch-style tail measures


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
    """Largest relative ES gap, ES gap to its bound, VaR gap and whole-bucket ES gap."""

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


def es_check(fails: list[str], tag: str, dist: str, level: str, a: Agg) -> None:
    """Tolerance (b): the gap as a fraction of the deterministic bound."""
    tol = ES_TOL_FRACTION[dist]
    if a.ratio > tol * (1.0 + 1e-9):
        fails.append(f"{tag} {dist} {level}: gap/bound {a.ratio:.3f} > {tol}")


HDR = (
    f"{'dist':<10} {'level':<11} {'runs':>5} {'max gap/ES':>11} {'gap/bound':>10} "
    f"{'viol':>5} {'max VaR rel':>12} {'whole-bucket gap':>17}"
)


def row(dist, level, a: Agg):
    return (
        f"{dist:<10} {level:<11} {a.n:>5} {a.gap:>11.3e} {a.ratio:>10.3f} {a.viol:>5} "
        f"{a.var:>12.3e} {a.full:>17.3e}"
    )


def run_es(seeds: int, ns=(10_000, 100_000)) -> list[str]:
    fails: list[str] = []
    print(
        f"alpha={ALPHA!r} gamma={GAMMA!r} bound alpha*mean|x| (= alpha*|ES| for a one-signed tail)"
    )
    for n in ns:
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
                es_check(fails, f"es N={n}", dist, name, a)
        print(f"largest gap/ES over all: {worst:.3e}   alpha = {ALPHA:.3e}")
    return fails


def run_decay(seeds: int, lam: float = 0.999, ns=(10_000, 100_000)) -> list[str]:
    fails: list[str] = []
    print(
        f"lambda = {lam}; weights lam^(N-1-t) on the row clock; exact = weighted tail mean, "
        f"same fractional mass q*sum(w)"
    )
    for n in ns:
        print(f"\nN = {n}, seeds = {seeds}, n_eff = {1 / (1 - lam):.0f}")
        print(HDR)
        w = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        for dist in DISTRIBUTIONS:
            for name, side, q in LEVELS:
                a = Agg()
                for seed in range(seeds):
                    a.add(*level_pair(stream(dist, seed, n), w, side, q))
                print(row(dist, name, a))
                es_check(fails, f"decay N={n}", dist, name, a)
    # the batch weights equal the streaming recursion c <- lam*c + onehot(bucket), up to rounding
    print(
        "\nstreaming recursion (every bucket times lambda per row, then +1) vs the batch weights,"
    )
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
        d = np.max(np.abs(c - b) / b)
        print(f"  {dist:<10} buckets={len(u):>4}  max rel diff = {d:.2e}")
        if d > 1e-12:
            fails.append(f"decay recursion {dist}: {d:.2e} > 1e-12")
    return fails


def inv_of(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.searchsorted(u, k)


def run_window(seeds: int, n: int = 100_000, wsize: int = 5000, step: int = 5000) -> list[str]:
    fails: list[str] = []
    print(
        f"window W = {wsize}, N = {n}, query every {step} rows from the first full window; "
        f"counts removed exactly (integer subtract); seeds = {seeds}"
    )
    print(HDR + f" {'max live':>9}")
    for dist in DISTRIBUTIONS:
        aggs = {nm: Agg() for nm, _, _ in LEVELS}
        live_max = 0
        exact_counts = 0
        checks = 0
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
                        checks += 1
                        live = c > 0
                        live_max = max(live_max, int(live.sum()))
                        m = q * wsize
                        es_s, var_s, full_s, _ = tail_from(rp[live], c[live], m)
                        es_e, var_e, mabs = exact_tail(y, None, m)
                        sg = 1.0 if side == "lower" else -1.0
                        aggs[nm].add(
                            (sg * es_s, sg * var_s, sg * full_s), (sg * es_e, sg * var_e, mabs)
                        )
        for nm, _, _ in LEVELS:
            print(row(dist, nm, aggs[nm]) + f" {live_max:>9}")
            es_check(fails, "window", dist, nm, aggs[nm])
        print(
            f"  {dist}: windowed counts equal the bincount of the window at "
            f"{exact_counts} of {checks} checks"
        )
        if exact_counts != checks:  # tolerance (c): exact
            fails.append(
                f"window {dist}: counts differ at {checks - exact_counts} of {checks} checks"
            )
    return fails


def oracle_bins(s) -> dict:
    out = {}
    for sg, st in ((1, s._store), (-1, s._negative_store)):
        for idx, cnt in enumerate(st.bins):
            if cnt > 0:
                out[(sg, idx + st.offset)] = int(cnt)
    return out


def run_buckets(seeds: int, ns=(10_000, 100_000, 1_000_000), oracle_seeds: int = 5) -> list[str]:
    from ddsketch.ddsketch import BaseDDSketch
    from ddsketch.mapping import LogarithmicMapping
    from ddsketch.store import DenseStore

    fails: list[str] = []
    print(
        "live buckets (positive + negative + zero) of one signed sketch; bytes = live*16 + "
        f"{HEADER_BYTES} header"
    )
    for n in ns:
        print(f"\nN = {n}")
        print(
            f"{'dist':<10} {'min':>5} {'mean':>7} {'max':>5} {'max bytes':>10} "
            f"{'tail-0.99 buckets (mean)':>26}"
        )
        for dist in DISTRIBUTIONS:
            nb, tb = [], []
            for seed in range(seeds if n < 1_000_000 else 2):
                x = stream(dist, seed, n)
                u, c = sketch_of(x)
                nb.append(len(u))
                # buckets that hold the top 1% of the mass
                cum = np.cumsum(c[::-1])
                tb.append(int(np.searchsorted(cum, 0.01 * n)) + 1)
            print(
                f"{dist:<10} {min(nb):>5} {np.mean(nb):>7.1f} {max(nb):>5} "
                f"{max(nb) * BUCKET_BYTES + HEADER_BYTES:>10} {np.mean(tb):>26.1f}"
            )
    on = 100_000
    print(
        f"\nddsketch {version('ddsketch')} oracle (LogarithmicMapping(alpha), unbounded store), "
        f"N = {on}:"
    )
    print("  bucket counts compared with this module's, key by key; VaR within alpha of exact")
    mism = tot = 0
    worst = 0.0
    for dist in DISTRIBUTIONS:
        for seed in range(min(seeds, oracle_seeds)):
            x = stream(dist, seed, on)
            s = BaseDDSketch(LogarithmicMapping(ALPHA), DenseStore(), DenseStore(), 0.0)
            for v in x.tolist():
                s.add(v)
            mine = {}
            u, c = sketch_of(x)
            for k, cc in zip(u.tolist(), c.tolist(), strict=True):
                mine[(1 if k > 0 else -1, abs(k) - OFFSET)] = int(cc)
            theirs = oracle_bins(s)
            tot += len(mine)
            mism += sum(1 for k in set(mine) | set(theirs) if mine.get(k, 0) != theirs.get(k, 0))
            for p in (0.01, 0.05, 0.5, 0.95, 0.99):
                ex = float(np.quantile(x, p, method="inverted_cdf"))
                worst = max(worst, abs(s.get_quantile_value(p) - ex) / abs(ex))
    print(
        f"  buckets compared {tot}, mismatching {mism}; largest ddsketch quantile rel. error "
        f"vs inverted_cdf {worst:.3e} (alpha {ALPHA:.3e})"
    )
    if mism:
        fails.append(f"buckets: {mism} of {tot} bucket counts differ from ddsketch")
    if worst > ALPHA * (1 + 1e-9):
        fails.append(f"buckets: ddsketch quantile error {worst:.3e} > alpha")
    return fails


def run_tails(seeds: int, only: str | None, quick: bool) -> list[str]:
    fails: list[str] = []
    parts = {
        "es": lambda: run_es(seeds, (10_000,) if quick else (10_000, 100_000)),
        "decay": lambda: run_decay(seeds, ns=(10_000,) if quick else (10_000, 100_000)),
        "window": lambda: run_window(seeds, n=20_000 if quick else 100_000),
        "buckets": lambda: run_buckets(
            seeds,
            ns=(10_000, 100_000) if quick else (10_000, 100_000, 1_000_000),
            oracle_seeds=1 if quick else 5,
        ),
    }
    for name, fn in parts.items():
        if only in (None, name):
            print(f"\n=== tails: {name} ===")
            fails += fn()
    return fails


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
        for mi, wi in zip(m[1:], w[1:], strict=True):
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
            return (
                float(self.hi - (self.hi - m[-1]) * r / (w[-1] / 2.0))
                if w[-1] > 1
                else float(m[-1])
            )
        centers = np.cumsum(w) - w / 2.0
        i = int(np.searchsorted(centers, t, side="right")) - 1
        f = (t - centers[i]) / (centers[i + 1] - centers[i])
        return float(m[i] + f * (m[i + 1] - m[i]))

    @property
    def state_bytes(self) -> int:
        return 16 * len(self.means)


PS = (0.01, 0.05, 0.5, 0.95, 0.99)
COMPRESSIONS = (100, 200)
IMPLS = ("mine", "pytdigest", "crick", "tdigest")
PAIRS = [(a, b) for i, a in enumerate(IMPLS) for b in IMPLS[i + 1 :]]


def rank_of(sorted_x: np.ndarray, q: float) -> float:
    return float(np.searchsorted(sorted_x, q, side="right")) / len(sorted_x)


def k1_span(weights, delta: float) -> float:
    """Largest k1-span (delta/(2 pi) asin(2q-1)) of a centroid list; the size rule is span <= 1."""
    w = np.asarray(weights, dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(w)]) / w.sum()
    k = delta / (2.0 * math.pi) * np.arcsin(np.clip(2.0 * c - 1.0, -1.0, 1.0))
    return float(np.max(np.diff(k)))


def td_unit(job):
    """One stream: the quantiles' ranks, centroid counts and k1-spans of every implementation."""
    import crick
    import pytdigest
    import tdigest

    dist, seed, n, use_tdigest = job
    x = stream(dist, seed, n)
    sx = np.sort(x)
    out = {}
    for c in COMPRESSIONS:
        md = MergingDigest(c)
        md.update_many(x)
        pt = pytdigest.TDigest.compute(x, compression=c)
        cr = crick.TDigest(compression=c)
        cr.update(x)
        qs = {
            "mine": [md.quantile(p) for p in PS],
            "pytdigest": [pt.inverse_cdf(p) for p in PS],
            "crick": [float(cr.quantile(p)) for p in PS],
        }
        ptc, crc = pt.get_centroids(), cr.centroids()
        cent = {"mine": len(md.means), "pytdigest": len(ptc), "crick": len(crc)}
        span = {
            "mine": k1_span(md.weights, c),
            "pytdigest": k1_span(ptc[:, 1], c),
            "crick": k1_span(crc["weight"], c),
        }
        if use_tdigest:  # the pure-Python oracle takes 3 to 6 s a stream
            td = tdigest.TDigest(delta=1.0 / c)
            td.batch_update(x)
            qs["tdigest"] = [td.percentile(100 * p) for p in PS]
            cent["tdigest"] = len(td)
            span["tdigest"] = k1_span([d["c"] for d in td.centroids_to_list()], c)
        out[c] = {
            "rank": {im: [rank_of(sx, q) for q in v] for im, v in qs.items()},
            "cent": cent,
            "span": span,
        }
    return dist, seed, out


def run_tdigest(seeds: int, td_seeds: int, n: int = 100_000, workers: int = 4) -> list[str]:
    print(
        f"rank error |F(q_hat) - p|, F the empirical CDF of the stream; N = {n}, seeds = {seeds}, "
        f"tdigest on the first {td_seeds} seeds, workers = {workers}"
    )
    print(
        "  mine = Dunning-Ertl merging digest, k1, buffer 8 delta (this module); "
        f"pytdigest {version('pytdigest')}; crick {version('crick')}; tdigest {version('tdigest')} "
        "with delta = 1/compression, K = 25 (an older, tree-based algorithm)"
    )
    jobs = [(d, s, n, s < td_seeds) for s in range(seeds) for d in DISTRIBUTIONS]
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(td_unit, jobs), 1):
                results.append(r)
                if i % 8 == 0 or i == len(jobs):
                    print(f"  ... {i}/{len(jobs)} streams, {time.time() - t0:.0f} s", flush=True)
    else:
        results = [td_unit(j) for j in jobs]
    results.sort(key=lambda r: (r[1], DISTRIBUTIONS.index(r[0])))
    fails: list[str] = []
    for c in COMPRESSIONS:
        err = {im: {p: 0.0 for p in PS} for im in IMPLS}
        cent = {im: [] for im in IMPLS}
        span = {im: 0.0 for im in IMPLS}
        pair = {pr: 0.0 for pr in PAIRS}
        perdist = {im: {d: 0.0 for d in DISTRIBUTIONS} for im in IMPLS}
        for dist, _seed, out in results:
            o = out[c]
            for im, ranks in o["rank"].items():
                cent[im].append(o["cent"][im])
                span[im] = max(span[im], o["span"][im])
                for p, r in zip(PS, ranks, strict=True):
                    e = abs(r - p)
                    err[im][p] = max(err[im][p], e)
                    perdist[im][dist] = max(perdist[im][dist], e)
            for a, b in PAIRS:
                if a in o["rank"] and b in o["rank"]:
                    d = max(abs(x - y) for x, y in zip(o["rank"][a], o["rank"][b], strict=True))
                    pair[(a, b)] = max(pair[(a, b)], d)
        print(f"\ncompression {c}: largest rank error over 4 distributions x seeds, by p")
        print(
            f"{'impl':<10} "
            + " ".join(f"p={p:<6}" for p in PS)
            + f" {'max':>9} {'centroids min/mean/max':>24} {'bytes (max)':>12} {'max k1-span':>12}"
        )
        for im in IMPLS:
            mx = max(err[im].values())
            ce = cent[im]
            print(
                f"{im:<10} "
                + " ".join(f"{err[im][p]:<8.2e}" for p in PS)
                + f" {mx:>9.2e} {min(ce):>7}/{np.mean(ce):>7.1f}/{max(ce):>7} {16 * max(ce):>12}"
                f" {span[im]:>12.2f}"
            )
        print(
            "  largest rank difference between two implementations' quantile (same stream and p):"
        )
        for (a, b), v in pair.items():
            print(f"    {a:<9} vs {b:<9} {v:.2e}")
        print(
            "  per distribution (largest over p):",
            {d: {im: f"{perdist[im][d]:.1e}" for im in IMPLS} for d in DISTRIBUTIONS},
        )
        mx = max(err["mine"].values())
        if mx > TD_TOL[c]:
            fails.append(
                f"tdigest compression {c}: rank error {mx:.2e} > tolerance {TD_TOL[c]:.2e}"
            )
    return fails


# ------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["psquared", "tails", "tdigest", "all", "oracles", "state"])
    ap.add_argument(
        "--quick", action="store_true", help="small streams, few seeds; checks the tolerances"
    )
    ap.add_argument("--n", type=int, default=None, help="psquared stream length (default 1000)")
    ap.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="seeds (psquared 50, tails 20, tdigest 20; quick 3, 2, 1)",
    )
    ap.add_argument(
        "--td-seeds",
        type=int,
        default=None,
        help="seeds through the pure-Python tdigest (20; quick 1)",
    )
    ap.add_argument(
        "--trajectory", action="store_true", help="psquared: compare after every update"
    )
    ap.add_argument("--only", choices=["es", "decay", "window", "buckets"], default=None)
    ap.add_argument("--workers", type=int, default=None, help="tdigest processes (4)")
    a = ap.parse_args()
    if a.what == "oracles":
        for r in oracle_table():
            print(" | ".join(f"{k}={v}" for k, v in r.items()))
        return 0
    if a.what == "state":
        print_state()
        return 0
    q = a.quick
    t0 = time.time()
    fails: list[str] = []
    if a.what in ("psquared", "all"):
        print("=== psquared ===")
        fails += run_psquared(a.n or 1000, a.seeds or (3 if q else 50), a.trajectory or q)
    if a.what in ("tails", "all"):
        fails += run_tails(a.seeds or (2 if q else 20), a.only, q)
    if a.what in ("tdigest", "all"):
        print("\n=== tdigest ===")
        fails += run_tdigest(
            a.seeds or (1 if q else 20),
            a.td_seeds if a.td_seeds is not None else (1 if q else 20),
            n=100_000,
            workers=a.workers or 4,
        )
    print(f"\n{'FAIL' if fails else 'PASS'} ({time.time() - t0:.0f} s)")
    for f in fails:
        print("  FAILED:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
