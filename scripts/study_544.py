#!/usr/bin/env python
"""Study 544: do rheon's streaming tail measures agree with independent oracles?

Part 1: the oracle search and a numpy port of P-squared (Jain and Chlamtac 1985)
checked against river's ``stats.Quantile``.  Python only; nothing here touches
the Rust core.  Every generator is seeded and no data file is written.

    python scripts/study_544.py oracles            # package, version, licence, URL
    python scripts/study_544.py p2 [--n 1000] [--seeds 50] [--trajectory]
    python scripts/study_544.py state              # bytes of the P-squared state

Run it in a venv holding the oracles (this study used ~/study544-venv, not the
project's .venv, so the project environment stays untouched):

    uv pip install river pytdigest tdigest ddsketch crick datasketches numpy
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, metadata, version

import numpy as np

ORACLES = ["river", "pytdigest", "tdigest", "ddsketch", "crick", "datasketches", "numpy", "scipy"]
QUANTILES = (0.05, 0.5, 0.95)
DISTRIBUTIONS = ("normal", "lognormal", "student_t3", "pareto2.5")


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


def run_p2(n: int, seeds: int, trajectory: bool) -> int:
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
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["oracles", "p2", "state"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--trajectory", action="store_true", help="compare after every update")
    a = ap.parse_args()
    if a.what == "oracles":
        for row in oracle_table():
            print(" | ".join(f"{k}={v}" for k, v in row.items()))
        return 0
    if a.what == "state":
        for k, v in state_bytes().items():
            print(f"{k:<28}{v}")
        return 0
    return run_p2(a.n, a.seeds, a.trajectory)


if __name__ == "__main__":
    sys.exit(main())
