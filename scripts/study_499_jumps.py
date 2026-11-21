"""Study 499b: oracles for bipower variation and the jump tests (PLAN task 499).

Python only, plus R through rpy2.  Nothing in the package or the Rust crates
is touched.  The record is ``docs/records/STUDY-499-JUMPS.md``.

Run one part at a time, from the study's own environment::

    uv venv .venv-499 --python 3.12
    uv pip install --python .venv-499 numpy scipy polars pyarrow mpmath hfvol \\
        mfe-toolbox realized-library hffe tick "rpy2==3.5.17"
    # R 4.3 (apt: r-base-core r-base-dev r-cran-xts r-cran-zoo r-cran-rcpp ...),
    # then in R:  install.packages("highfrequency", lib = "/root/Rlib")
    # and hf.econometrics from a git checkout (see the record, section 2)
    .venv-499/bin/python scripts/study_499_jumps.py part1

Parts (each writes ``.cache/study_499_jumps/partN.json`` and a table on
stdout; a finished part is not recomputed without ``--force``):

* ``part1``: the oracle inventory (versions, licences), the paper's definitions
  (``mu_p``, bipower variation, tripower quarticity), and every candidate's
  bipower variation and tripower quarticity on synthetic returns without and
  with planted jumps and on the cached Binance data.

Every candidate is compared with the paper's definition evaluated in 40-digit
arithmetic (mpmath) on the very float64 returns the candidate sees, so a gap
in "ulps" is a gap to the exact value of the formula, and the paper's formula
in float64 numpy is itself one of the rows.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "study_499_jumps"
MICRO = ROOT / ".cache" / "microstructure"

os.environ.setdefault("R_HOME", "/usr/lib/R")
os.environ.setdefault("R_LIBS", os.environ.get("STUDY_R_LIBS", "/root/Rlib"))

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# The paper's definitions
# --------------------------------------------------------------------------
#
# Barndorff-Nielsen and Shephard (2004), "Power and bipower variation with
# stochastic volatility and jumps", JFEC 2(1):
#   mu_p = E|Z|^p = 2^(p/2) Gamma((p+1)/2) / Gamma(1/2),  Z ~ N(0,1)
#   realized bipower variation  BV = mu_1^-2 sum_{j=2}^n |r_j| |r_{j-1}|
#   (their {y}^[1,1]; the factor (1/n)^(1-(r+s)/2) is 1 for r = s = 1)
# Barndorff-Nielsen and Shephard (2006), "Econometrics of testing for jumps in
# financial economics using bipower variation", JFEC 4(1):
#   tripower quarticity  TQ = n mu_{4/3}^-3 sum_{j=3}^n |r_j|^{4/3} |r_{j-1}|^{4/3} |r_{j-2}|^{4/3}
#   (Delta^-1 with Delta = 1/n; no small-sample factor).
# Neither paper applies a small-sample factor such as n/(n-1) or n/(n-2).


def mu_p(p: float) -> float:
    """E|Z|^p in float64: 2^(p/2) Gamma((p+1)/2) / Gamma(1/2)."""
    return 2.0 ** (p / 2.0) * math.gamma((p + 1.0) / 2.0) / math.gamma(0.5)


def bv_paper(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
    a = np.abs(r)
    return float(np.sum(a[1:] * a[:-1]) / mu_p(1.0) ** 2)


def tpq_paper(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
    a = np.abs(r) ** (4.0 / 3.0)
    n = len(r)
    return float(n * np.sum(a[2:] * a[1:-1] * a[:-2]) / mu_p(4.0 / 3.0) ** 3)


def exact_reference(r: np.ndarray) -> dict:
    """The paper's formulas for these exact float64 returns, in 40 digits."""
    import mpmath as mp

    mp.mp.dps = 40
    n = len(r)
    a = [abs(mp.mpf(float(x))) for x in r]
    mu1 = mp.sqrt(2 / mp.pi)
    mu43 = mp.mpf(2) ** (mp.mpf(2) / 3) * mp.gamma(mp.mpf(7) / 6) / mp.gamma(mp.mpf(1) / 2)
    bv = mp.fsum(a[j] * a[j - 1] for j in range(1, n)) / mu1**2
    p = [x ** (mp.mpf(4) / 3) for x in a]
    tq = n * mp.fsum(p[j] * p[j - 1] * p[j - 2] for j in range(2, n)) / mu43**3
    rv = mp.fsum(x * x for x in a)
    return {"bv": bv, "tpq": tq, "rv": rv}


def exact_constants() -> dict:
    import mpmath as mp

    mp.mp.dps = 40
    mu1 = mp.sqrt(2 / mp.pi)
    mu43 = mp.mpf(2) ** (mp.mpf(2) / 3) * mp.gamma(mp.mpf(7) / 6) / mp.gamma(mp.mpf(1) / 2)
    # the definition E|Z|^p by quadrature, as a check of the closed form
    q = lambda p: 2 * mp.quad(lambda z: z**p * mp.npdf(z), [0, mp.inf])  # noqa: E731
    return {"mu1": mu1, "mu43": mu43, "mu1_quad": q(1), "mu43_quad": q(mp.mpf(4) / 3)}


def ulps(value: float, ref) -> float:
    """|value - ref| in ulps of float(ref); ref is an mpmath number."""
    import mpmath as mp

    f = float(ref)
    u = math.ulp(f)
    return float(abs(mp.mpf(float(value)) - ref) / mp.mpf(u))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def synth(n: int, seed: int, jumps: bool = False):
    """One day of n returns: a stochastic-volatility diffusion, optionally with five jumps.

    sd of a return = s0 sqrt(1/n) u(t) exp(h/2) with s0 = 1% a day, u an intraday
    U-shape and h a mean-reverting log-volatility factor (AR(1), half-life n/10
    returns).  The jumps sit at fixed fractions of the day with sizes
    (+8, -10, +6, +12, -7) local standard deviations.  The same seed gives the
    same diffusion with and without the jumps.
    """
    rng = np.random.default_rng(1000 * n + seed)
    t = (np.arange(n) + 0.5) / n
    u = 1.0 + 0.6 * (2 * t - 1) ** 2
    u /= np.sqrt(np.mean(u**2))
    phi = 0.5 ** (10.0 / n)
    e = rng.standard_normal(n)
    h = np.empty(n)
    h[0] = 0.0
    for j in range(1, n):
        h[j] = phi * h[j - 1] + math.sqrt(1 - phi * phi) * 0.5 * e[j]
    sd = 0.01 * math.sqrt(1.0 / n) * u * np.exp(0.5 * h)
    r = sd * rng.standard_normal(n)
    idx = np.array([], dtype=int)
    size = np.array([])
    if jumps:
        idx = np.array([int(f * n) for f in (0.1, 0.3, 0.5, 0.7, 0.9)])
        size = np.array([8.0, -10.0, 6.0, 12.0, -7.0]) * sd[idx]
        r = r.copy()
        r[idx] += size
    return r, idx, size


def prices_from(r: np.ndarray) -> np.ndarray:
    """Prices whose log returns are r up to rounding (p0 = 100)."""
    return 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def real_series() -> dict[str, np.ndarray]:
    """Prices of the cached Binance data: BTCUSDT 1-minute closes (10 days) and ETCUSDT trades."""
    import polars as pl

    out: dict[str, np.ndarray] = {}
    days = sorted(CACHE.parent.glob("BTCUSDT-1m-2024-01-*.parquet"))
    for d in days:
        df = pl.read_parquet(d)
        out[f"BTCUSDT 1m {d.stem[-10:]}"] = df["close"].to_numpy().astype(np.float64)
    f = MICRO / "ETCUSDT-2024-01-02.parquet"
    if f.exists():
        tr = pl.read_parquet(f).filter(pl.col("kind") == "trade").select("ts", "price").sort("ts")
        for every, label in (("10s", "10 s"), ("1m", "1 min")):
            g = (
                tr.group_by_dynamic("ts", every=every)
                .agg(pl.col("price").last())
                .sort("ts")
                .upsample("ts", every=every)
                .with_columns(pl.col("price").forward_fill())
            )
            out[f"ETCUSDT trades {label} 2024-01-02"] = g["price"].to_numpy().astype(np.float64)
    return out


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------
# Each candidate: name, quantity, a function of (returns, prices) -> float, and
# the factor f(n) its source applies on top of the paper's definition
# (read from its source; n is the number of returns).  A candidate whose
# residual after f(n) is a few ulps follows the paper up to that convention.


def _r():
    """rpy2 with highfrequency and hf.econometrics loaded."""
    import rpy2.robjects as ro

    ro.r("suppressMessages({library(highfrequency); library(hf.econometrics)})")
    return ro


def _shim_scipy() -> None:
    """realized-library 0.1.2 imports scipy.integrate.simps and trapz, removed in scipy 1.14.

    The shim aliases them to their successors (simpson, trapezoid) before the import; the
    library's own code is untouched and neither name is used by the quantities studied here.
    """
    import scipy.integrate as si

    if not hasattr(si, "simps"):
        si.simps = si.simpson  # type: ignore[attr-defined]
    if not hasattr(si, "trapz"):
        si.trapz = si.trapezoid  # type: ignore[attr-defined]


def candidates() -> list[dict]:
    _shim_scipy()
    import hfvol
    import mfe.realized.quarticity as mq
    import mfe.realized.variance as mv
    import realized_library.estimators.variance.bipower_variation as rl_bpv
    import realized_library.estimators.variance.multipower_variation as rl_mpv
    from hffe.stock import Stock

    ro = _r()
    fv = ro.FloatVector

    def hf_r(fn):
        return lambda r, p: float(ro.r[fn](fv(r))[0])

    def hfe_bv(adjust):
        f = ro.r("function(p, a) hf.econometrics::iv_bipower(p, adjust = a)@value")
        return lambda r, p: float(f(fv(np.log(p)), ro.BoolVector([adjust]))[0])  # takes log prices

    one = lambda n: 1.0  # noqa: E731
    C = []
    add = lambda name, q, fn, fac=one, note="": C.append(  # noqa: E731
        {"name": name, "q": q, "fn": fn, "factor": fac, "note": note}
    )
    # bipower variation
    add("paper (numpy, float64)", "bv", lambda r, p: bv_paper(r))
    add("hfvol 0.1.0 bipower_variation", "bv", lambda r, p: hfvol.bipower_variation(r))
    add(
        "mfe-toolbox realized_bipower_variation .value",
        "bv",
        lambda r, p: float(mv.realized_bipower_variation(r).value),
    )
    add(
        "mfe-toolbox ... .debiased_value",
        "bv",
        lambda r, p: float(mv.realized_bipower_variation(r).debiased_value),
        lambda n: (n - 1) / (n - 2),
        "m/(m-1) with m = n-1 products",
    )
    add(
        "realized-library bipower_variation (default)",
        "bv",
        lambda r, p: float(rl_bpv.compute(p)),
        lambda n: n / (n - 1),
        "n/(n-1)",
    )
    add(
        "realized-library bipower_variation (no bias factor)",
        "bv",
        lambda r, p: float(rl_bpv.compute(p, correct_scaling_bias=False)),
    )
    add(
        "realized-library multipower_variation m=2 r=2 (default)",
        "bv",
        lambda r, p: float(rl_mpv.compute(p, m=2, r=2)),
        lambda n: n / (n - 1),
        "n/(n-1)",
    )
    add("hffe 0.0.13 Stock.getBV", "bv", lambda r, p: float(Stock.getBV(r[:, None])[0]))
    add("highfrequency 1.0.3 rBPCov", "bv", hf_r("rBPCov"))
    add("hf.econometrics iv_bipower (adjust = FALSE)", "bv", hfe_bv(False))
    add(
        "hf.econometrics iv_bipower (adjust = TRUE)",
        "bv",
        hfe_bv(True),
        lambda n: n / (n - 1),
        "n/(n-1)",
    )
    # tripower quarticity
    add("paper (numpy, float64)", "tpq", lambda r, p: tpq_paper(r))
    add(
        "mfe-toolbox realized_tripower_quarticity",
        "tpq",
        lambda r, p: float(mq.realized_tripower_quarticity(r).value),
        lambda n: 1.0 / (n - 2),
        "n * mean() where the paper has n * sum()",
    )
    add(
        "realized-library multipower_variation m=3 r=4 (default)",
        "tpq",
        lambda r, p: float(rl_mpv.compute(p, m=3, r=4)),
        lambda n: n / (n - 2),
        "n/(n-2)",
    )
    add(
        "realized-library multipower_variation m=3 r=4 (no bias factor)",
        "tpq",
        lambda r, p: float(rl_mpv.compute(p, m=3, r=4, correct_scaling_bias=False)),
    )
    add("highfrequency 1.0.3 rTPQuar", "tpq", hf_r("rTPQuar"), lambda n: n / (n - 2), "n/(n-2)")
    return C


def inventory() -> list[dict]:
    rows = []
    for dist in (
        "hfvol",
        "mfe-toolbox",
        "realized-library",
        "hffe",
        "highfrequency",
        "tick",
        "numpy",
        "scipy",
        "mpmath",
        "rpy2",
    ):
        m = md.metadata(dist)
        lic = m.get("License-Expression") or m.get("License") or ""
        for c in m.get_all("Classifier") or []:
            if not lic and c.startswith("License ::"):
                lic = c.split("::")[-1].strip()
        if dist == "scipy":
            lic = "BSD-3-Clause"
        if dist == "numpy":
            lic = "BSD-3-Clause"
        rows.append(
            {
                "name": f"PyPI {dist}",
                "version": md.version(dist),
                "license": lic.split("\n")[0][:40],
            }
        )
    ro = _r()
    for pkg in ("highfrequency", "hf.econometrics", "yuima"):
        try:
            d = ro.r(f'packageDescription("{pkg}")')
        except Exception:  # noqa: BLE001
            continue
        names = list(d.names)

        def g(k, d=d, names=names):
            return str(d[names.index(k)][0]) if k in names else ""

        rows.append(
            {
                "name": f"CRAN/R {pkg}",
                "version": g("Version"),
                "license": g("License"),
                "extra": g("Date/Publication") or g("Packaged")[:30],
            }
        )
    rows.append({"name": "R", "version": str(ro.r("R.version.string")[0]), "license": "GPL"})
    return rows


# --------------------------------------------------------------------------
# Part 1
# --------------------------------------------------------------------------


def part1(force: bool) -> None:
    out = CACHE / "part1.json"
    if out.exists() and not force:
        print(f"{out} exists; not recomputing (use --force)")
        print(json.dumps(json.loads(out.read_text())["summary"], indent=1)[:200])
        return
    t0 = time.time()
    res: dict = {"inventory": inventory()}

    # constants
    ex = exact_constants()
    import hfvol.realized as hr

    _shim_scipy()
    import mfe.realized.quarticity as mq
    from realized_library._utils.std_norm_dist_moments import mu_x

    consts = {
        "mu_1 exact": str(ex["mu1"])[:30],
        "mu_1 quadrature vs closed form (abs)": float(abs(ex["mu1_quad"] - ex["mu1"])),
        "mu_{4/3} exact": str(ex["mu43"])[:30],
        "mu_{4/3} quadrature vs closed form (abs)": float(abs(ex["mu43_quad"] - ex["mu43"])),
        "ulps: paper mu_1 (math.gamma)": ulps(mu_p(1.0), ex["mu1"]),
        "ulps: hfvol MU1": ulps(hr.MU1, ex["mu1"]),
        "ulps: mfe _MU1": ulps(float(mq._MU1), ex["mu1"]),
        "ulps: realized-library mu_x(1)": ulps(float(mu_x(1)), ex["mu1"]),
        "ulps: paper mu_{4/3} (math.gamma)": ulps(mu_p(4 / 3), ex["mu43"]),
        "ulps: mfe _MU43": ulps(float(mq._MU43), ex["mu43"]),
        "ulps: realized-library mu_x(4/3)": ulps(float(mu_x(4 / 3)), ex["mu43"]),
        "ulps: highfrequency 1/rTPQuar constant": None,
    }
    ro = _r()
    c = float(ro.r("(gamma(0.5)/(2^(2/3)*gamma(7/6)))^3")[0])
    import mpmath as mp

    consts["ulps: highfrequency 1/rTPQuar constant"] = ulps(c, 1 / ex["mu43"] ** 3)
    res["constants"] = consts

    # series
    series: list[dict] = []
    for n in (78, 390, 2340, 23400):
        for seed in range(3):
            r0, _, _ = synth(n, seed, False)
            r1, idx, _ = synth(n, seed, True)
            for kind, r in (("synthetic, no jumps", r0), ("synthetic, 5 jumps", r1)):
                p = prices_from(r)
                series.append(
                    {
                        "group": f"{kind}, n={n}",
                        "label": f"{kind} n={n} seed={seed}",
                        "p": p,
                        "r": np.diff(np.log(p)),
                        "partner": r0 if "jumps" in kind else None,
                    }
                )
    for label, p in real_series().items():
        grp = label.split(" 2024")[0]
        series.append(
            {
                "group": f"Binance {grp}",
                "label": label,
                "p": p,
                "r": np.diff(np.log(p)),
                "partner": None,
            }
        )
    print(f"{len(series)} series; setup {time.time() - t0:.1f} s")

    # reference values
    t1 = time.time()
    for s in series:
        s["ref"] = exact_reference(s["r"])
    print(f"exact references {time.time() - t1:.1f} s")

    cands = candidates()
    rows = []  # (candidate, quantity, group) -> list of (n, raw_ulps, factor_ulps, rel)
    t2 = time.time()
    for cd in cands:
        for s in series:
            n = len(s["r"])
            ref = s["ref"][cd["q"]]
            try:
                v = cd["fn"](s["r"], s["p"])
                err = None
            except Exception as e:  # noqa: BLE001
                v, err = float("nan"), f"{type(e).__name__}: {e}"[:80]
            if err is None:
                raw = ulps(v, ref)
                fac = cd["factor"](n)
                adj = ulps(v, ref * mp.mpf(fac)) if fac != 1.0 else raw
                rel = float(abs(mp.mpf(v) - ref) / ref)
                ratio = float(mp.mpf(v) / ref)
            else:
                raw = adj = rel = ratio = float("nan")
            rows.append(
                {
                    "cand": cd["name"],
                    "q": cd["q"],
                    "group": s["group"],
                    "label": s["label"],
                    "n": n,
                    "raw_ulps": raw,
                    "adj_ulps": adj,
                    "rel": rel,
                    "ratio": ratio,
                    "factor_note": cd["note"],
                    "err": err,
                    "v": v,
                }
            )
    print(f"candidates {time.time() - t2:.1f} s")
    res["rows"] = rows

    # candidates with no factor in their source, against one another (ulps of the larger value)
    pw = {}
    for q in ("bv", "tpq"):
        nofac = [
            cd["name"]
            for cd in cands
            if cd["q"] == q and cd["factor"](100) == 1.0 and not cd["name"].startswith("paper")
        ]
        worst = (0.0, "", "")
        for lab in {r["label"] for r in rows}:
            vals = {
                r["cand"]: r["v"]
                for r in rows
                if r["label"] == lab and r["q"] == q and r["cand"] in nofac
            }
            for a in vals:
                for b in vals:
                    if a < b:
                        d = abs(vals[a] - vals[b]) / math.ulp(max(vals[a], vals[b]))
                        if d > worst[0]:
                            worst = (d, a, b)
        pw[q] = {"candidates": nofac, "max_ulps": worst[0], "pair": worst[1:]}
    res["pairwise"] = pw

    # jump robustness of the paper's definitions: the same diffusion with and without the jumps
    rob = []
    for n in (390, 2340, 23400):
        for seed in range(3):
            r0, idx, size = synth(n, seed, False)
            r1, _, _ = synth(n, seed, True)
            rob.append(
                {
                    "n": n,
                    "seed": seed,
                    "rv_ratio": float(np.sum(r1**2) / np.sum(r0**2)),
                    "bv_ratio": bv_paper(r1) / bv_paper(r0),
                    "tpq_ratio": tpq_paper(r1) / tpq_paper(r0),
                    "jump_sq": float(np.sum(size**2)) if False else float(np.sum((r1 - r0) ** 2)),
                    "rv_gain": float(np.sum(r1**2) - np.sum(r0**2)),
                    "bv_gain": bv_paper(r1) - bv_paper(r0),
                }
            )
    res["robustness"] = rob

    # scale check: a quarticity estimator must be of the order of the realized quarticity
    # RQ = (n/3) sum r^4 on a series without jumps (both estimate int sigma^4)
    sc = []
    for s in series:
        if s["group"] != "synthetic, no jumps, n=23400":
            continue
        n = len(s["r"])
        rq = n / 3.0 * float(np.sum(s["r"] ** 4))
        for cd in cands:
            if cd["q"] == "tpq":
                sc.append(
                    {
                        "cand": cd["name"],
                        "label": s["label"],
                        "tpq_over_rq": cd["fn"](s["r"], s["p"]) / rq,
                    }
                )
    res["scale"] = sc
    res["series"] = [{"group": s["group"], "label": s["label"], "n": len(s["r"])} for s in series]
    res["summary"] = {"seconds": time.time() - t0}
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"part1 done in {time.time() - t0:.1f} s -> {out}")
    print_part1(res)


def print_part1(res: dict) -> None:
    print("\n## inventory")
    for r in res["inventory"]:
        print(f"| {r['name']} | {r['version']} | {r['license']} | {r.get('extra', '')} |")
    print("\n## constants")
    for k, v in res["constants"].items():
        print(f"| {k} | {v} |")
    import collections

    agg: dict = collections.OrderedDict()
    for r in res["rows"]:
        agg.setdefault((r["q"], r["cand"]), []).append(r)
    for q in ("bv", "tpq"):
        print(f"\n## {q}: per candidate, over all series")
        print(
            "| candidate | factor in source | max abs ratio-1 | max ulps raw "
            "| max ulps after factor | errors |"
        )
        for (qq, name), rs in agg.items():
            if qq != q:
                continue
            ok = [x for x in rs if x["err"] is None]
            nerr = len(rs) - len(ok)
            dev = max(abs(x["ratio"] - 1) for x in ok) if ok else float("nan")
            print(
                f"| {name} | {rs[0]['factor_note'] or '1'} | {dev:.3e} | "
                f"{max(x['raw_ulps'] for x in ok):.3g} | "
                f"{max(x['adj_ulps'] for x in ok):.3g} | {nerr} |"
                if ok
                else f"| {name} | | | | | {nerr} |"
            )
        print(f"\n### {q}: max ulps after the source's factor, by group")
        groups = sorted(
            {r["group"] for r in res["rows"] if r["q"] == q}, key=lambda g: (("Binance" in g), g)
        )
        print("| candidate | " + " | ".join(groups) + " |")
        for (qq, name), rs in agg.items():
            if qq != q:
                continue
            cells = []
            for g in groups:
                v = [x["adj_ulps"] for x in rs if x["group"] == g and x["err"] is None]
                cells.append(f"{max(v):.3g}" if v else "-")
            print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n## pairwise, candidates with no factor in their source")
    for q, v in res["pairwise"].items():
        print(
            f"| {q} | {v['max_ulps']:.3g} ulps | {v['pair']} | {len(v['candidates'])} candidates |"
        )
    print("\n## scale check: tripower quarticity / realized quarticity, no jumps, n = 23400")
    seen: dict = {}
    for x in res["scale"]:
        seen.setdefault(x["cand"], []).append(x["tpq_over_rq"])
    for k, v in seen.items():
        print(f"| {k} | " + ", ".join(f"{a:.4g}" for a in v) + " |")
    print("\n## jump robustness (paper definitions, same diffusion with / without five jumps)")
    for x in res["robustness"]:
        print(
            f"n={x['n']} seed={x['seed']}: RV x{x['rv_ratio']:.3f}  BV x{x['bv_ratio']:.4f}  "
            f"TPQ x{x['tpq_ratio']:.4f}  (jump sum of squares {x['jump_sq']:.3e}; "
            f"RV gain {x['rv_gain']:.3e}, "
            f"BV gain {x['bv_gain']:.3e})"
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=["part1"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)
    {"part1": part1}[a.part](a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
