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

* ``part2``: the Barndorff-Nielsen and Shephard (BNS) ratio and linear jump tests
  (R ``highfrequency`` ``BNSjumpTest``, R ``yuima`` ``bns.test``, ``realized-library``
  ``bns_test``, ``mfe-toolbox`` ``bns_jump_test``) and the Lee and Mykland (LM) statistic
  (``yuima`` ``lm.jumptest``, ``realized-library`` ``lm_test``, ``highfrequency``
  ``intradayJumpTest``), against the paper's definitions, with and without planted jumps and
  on the Binance data; the effect of ``mfe-toolbox``'s tripower-quarticity error on its test.

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


def build_series() -> list[dict]:
    """The 36 series of part 1 (and part 2): synthetic without and with jumps, and Binance."""
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
                        "jump_idx": idx if "jumps" in kind else np.array([], dtype=int),
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
                "jump_idx": np.array([], dtype=int),
            }
        )
    return series


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
    series = build_series()
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


# --------------------------------------------------------------------------
# Part 2: the BNS tests and the Lee and Mykland statistic
# --------------------------------------------------------------------------
#
# Barndorff-Nielsen and Shephard (2006; read here in the Nuffield working paper of
# 2 September 2004, eqs. 10-14), n returns of a unit interval, delta = 1/n,
# vartheta = pi^2/4 + pi - 5, BV the part-1 bipower variation (mu_1^-2 scaled) and
# QP = n mu_1^-4 sum_{j=4}^n |r_j||r_{j-1}||r_{j-2}||r_{j-3}| the quadpower quarticity:
#   G (linear)   = delta^-1/2 (BV - RV) / sqrt(QP)            -> N(0, vartheta)
#   H (ratio)    = delta^-1/2 (BV/RV - 1) / sqrt(QP / BV^2)   -> N(0, vartheta)
#   J (adjusted) = the same with max(1, QP / BV^2) under the root
# all three are significantly NEGATIVE under jumps; dividing by sqrt(vartheta) gives N(0,1).
# The study's Z is the NEGATIVE of the paper's statistic over sqrt(vartheta), so that it is
# positive for a jump (the sign of R highfrequency, yuima, realized-library):
#   Z_lin = sqrt(n) (RV - BV) / sqrt(vartheta Q)
#   Z_rat = sqrt(n) (1 - BV/RV) / sqrt(vartheta Q / BV^2),   Z_adj with max(1, Q/BV^2)
# Q is the paper's quadpower QP, or the tripower TQ of part 1 (Huang and Tauchen's choice and
# the default of R highfrequency).  No small-sample factor anywhere.
#
# Lee and Mykland (2008, RFS 21(6)), K the window, c = sqrt(2/pi), n returns:
#   L_i = |r_i| / sigma_i,  sigma_i^2 = 1/(K-2) sum_{j=i-K+2}^{i-1} |r_j||r_{j-1}|
#   (the current return is excluded from the window)
#   C_n = sqrt(2 log n)/c - (log pi + log log n) / (2 c sqrt(2 log n)),  S_n = 1/(c sqrt(2 log n))
#   T = (max_i |L_i| - C_n) / S_n  -> Gumbel under H0 (P(xi <= x) = exp(-e^-x));
#   reject at level alpha when T > beta* = -log(-log(1 - alpha)),   alpha = 5%: 2.970.
THETA = math.pi**2 / 4 + math.pi - 5


def _mp_prims(r: np.ndarray) -> dict:
    """Exact (40 digit) RV, BV, TQ and QP of the float64 returns r."""
    import mpmath as mp

    mp.mp.dps = 40
    n = len(r)
    a = [abs(mp.mpf(float(x))) for x in r]
    mu1 = mp.sqrt(2 / mp.pi)
    mu43 = mp.mpf(2) ** (mp.mpf(2) / 3) * mp.gamma(mp.mpf(7) / 6) / mp.gamma(mp.mpf(1) / 2)
    p = [x ** (mp.mpf(4) / 3) for x in a]
    return {
        "n": n,
        "rv": mp.fsum(x * x for x in a),
        "bv": mp.fsum(a[j] * a[j - 1] for j in range(1, n)) / mu1**2,
        "tp": n * mp.fsum(p[j] * p[j - 1] * p[j - 2] for j in range(2, n)) / mu43**3,
        "qp": n * mp.fsum(a[j] * a[j - 1] * a[j - 2] * a[j - 3] for j in range(3, n)) / mu1**4,
        "mu1": mu1,
    }


def bns_exact(pr: dict, spec: dict):
    """Z of one spec from the exact primitives (mpmath).

    spec: kind 'linear' | 'ratio' | 'adj'; quart 'qp' | 'tp'; qf(n) factor on Q; recip: the
    numerator is RV/BV - 1 instead of 1 - BV/RV; sqbug: the denominator is
    sqrt(vartheta (QPraw/BPVraw)^2) (realized-library 'ratio'); delta: the library's time step
    (default 1/n); q_delta: the quarticity carries delta^-1 in place of n; one: the constant in
    max(one, .) (default 1).
    """
    import mpmath as mp

    n = pr["n"]
    rv, bv = pr["rv"], pr["bv"]
    q = pr[spec["quart"]] * mp.mpf(spec.get("qf", lambda n: 1.0)(n))
    if spec.get("q_delta") and spec.get("delta"):  # the library's quarticity carries delta^-1
        q = q / (mp.mpf(spec["delta"]) * n)
    theta = mp.pi**2 / 4 + mp.pi - 5
    sd = 1 / mp.sqrt(mp.mpf(spec["delta"])) if spec.get("delta") else mp.sqrt(n)
    # delta: the library's time step (default 1/n); q_delta: its quarticity uses delta^-1, not n
    if spec["kind"] == "linear":
        return sd * (rv - bv) / mp.sqrt(theta * q)
    num = rv / bv - 1 if spec.get("recip") else 1 - bv / rv
    avar = q / bv**2
    if spec.get("sqbug"):
        avar = (pr["mu1"] ** 2 * pr["qp"] / bv) ** 2
    if spec["kind"] == "adj":
        avar = max(mp.mpf(spec.get("one", 1.0)), avar)
    return sd * num / mp.sqrt(theta * avar)


def bns_np(r: np.ndarray, kind: str = "adj", quart: str = "tp") -> float:
    """The paper's Z in float64 numpy (no factors)."""
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    a = np.abs(r)
    mu1 = math.sqrt(2 / math.pi)
    rv = float(np.sum(r * r))
    bv = float(np.sum(a[1:] * a[:-1])) / mu1**2
    if quart == "tp":
        q = tpq_paper(r)
    else:
        q = n * float(np.sum(a[3:] * a[2:-1] * a[1:-2] * a[:-3])) / mu1**4
    if kind == "linear":
        return math.sqrt(n) * (rv - bv) / math.sqrt(THETA * q)
    avar = q / bv**2
    if kind == "adj":
        avar = max(1.0, avar)
    return math.sqrt(n) * (1 - bv / rv) / math.sqrt(THETA * avar)


def bns_np_batch(
    R: np.ndarray, kind: str = "adj", quart: str = "tp", recip: bool = False, qf: float = 1.0
) -> np.ndarray:
    """bns_np over the rows of R (reps x n), vectorised; qf scales Q."""
    n = R.shape[1]
    a = np.abs(R)
    mu1 = math.sqrt(2 / math.pi)
    rv = np.sum(R * R, axis=1)
    bv = np.sum(a[:, 1:] * a[:, :-1], axis=1) / mu1**2
    if quart == "tp":
        b = a ** (4 / 3)
        q = n * np.sum(b[:, 2:] * b[:, 1:-1] * b[:, :-2], axis=1) / mu_p(4 / 3) ** 3
    else:
        q = n * np.sum(a[:, 3:] * a[:, 2:-1] * a[:, 1:-2] * a[:, :-3], axis=1) / mu1**4
    q = q * qf
    if kind == "linear":
        return math.sqrt(n) * (rv - bv) / np.sqrt(THETA * q)
    num = rv / bv - 1 if recip else 1 - bv / rv
    avar = q / bv**2
    if kind == "adj":
        avar = np.maximum(1.0, avar)
    return math.sqrt(n) * num / np.sqrt(THETA * avar)


def synth_fast(n: int, reps: int, seed: int, jumps: bool) -> np.ndarray:
    """reps days of n returns, as synth() (U-shape, log-OU volatility) but vectorised."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(777 * n + seed)
    t = (np.arange(n) + 0.5) / n
    u = 1.0 + 0.6 * (2 * t - 1) ** 2
    u /= np.sqrt(np.mean(u**2))
    phi = 0.5 ** (10.0 / n)
    e = rng.standard_normal((reps, n))
    h = lfilter([math.sqrt(1 - phi * phi) * 0.5], [1.0, -phi], e, axis=1)
    sd = 0.01 * math.sqrt(1.0 / n) * u[None, :] * np.exp(0.5 * h)
    r = sd * rng.standard_normal((reps, n))
    if jumps:
        idx = np.array([int(f * n) for f in (0.1, 0.3, 0.5, 0.7, 0.9)])
        r[:, idx] += np.array([8.0, -10.0, 6.0, 12.0, -7.0])[None, :] * sd[:, idx]
    return r


def bns_candidates() -> list[dict]:
    """Each BNS-test candidate with the conventions its source applies (read from the source)."""
    _shim_scipy()
    import mfe.realized.jumps as mj
    import realized_library.estimators.jump_detection.bns_test as rl_bns

    ro = _r()
    fv, bl = ro.FloatVector, ro.BoolVector
    hf = ro.r(
        "function(x, type, mx, iq) highfrequency::BNSjumpTest(x, type = type, max = mx,"
        " IQestimator = iq)$ztest"
    )
    yu = ro.r(
        "function(lp, r, type, adj) yuima::bns.test(yuima::setData(zoo::zoo(lp)), r = r,"
        " type = type, adj = adj)[[1]]$statistic"
    )
    day = 86400 * 10**9

    def ts_of(n):
        dt = int(round(day / n))
        return np.arange(n + 1, dtype=np.int64) * dt

    def rl(test):
        return lambda r, p: float(rl_bns.compute(p, ts_of(len(r)), test=test))

    def rl_delta(n):
        return int(round(day / n)) / day

    C: list[dict] = []

    def add(name, fn, spec, note="", delta_fn=None, q_delta=False):
        C.append(
            {
                "name": name,
                "fn": fn,
                "spec": spec,
                "note": note,
                "delta_fn": delta_fn,
                "q_delta": q_delta,
            }
        )

    nn = lambda n: n / (n - 2)  # noqa: E731
    for kind, typ, mx in (
        ("linear", "linear", False),
        ("ratio", "ratio", False),
        ("adj", "ratio", True),
    ):
        add(
            f"R highfrequency BNSjumpTest {typ}, max={'TRUE' if mx else 'FALSE'}",
            (lambda t, m: lambda r, p: float(hf(fv(r), t, bl([m]), "TP")[0]))(typ, mx),
            {"kind": kind, "quart": "tp", "qf": nn},
            "TQ with n/(n-2)",
        )
    for kind, typ, adj in (
        ("linear", "standard", True),
        ("ratio", "ratio", False),
        ("adj", "ratio", True),
    ):
        for rr, qk in (([1.0] * 4, "qp"), ([4 / 3] * 3, "tp")):
            add(
                f"R yuima bns.test {typ}, adj={'TRUE' if adj else 'FALSE'}, "
                f"r={'(1,1,1,1)' if qk == 'qp' else '(4/3 x3)'}",
                (lambda t, a, q: lambda r, p: float(yu(fv(np.log(p)), fv(q), t, bl([a]))[0]))(
                    typ, adj, rr
                ),
                {"kind": kind, "quart": qk},
                "sqrt of the number of prices (n+1), not of returns",
                delta_fn=lambda n: 1.0 / (n + 1),
            )
    add(
        "mfe-toolbox bns_jump_test (as shipped)",
        lambda r, p: float(mj.bns_jump_test(r).statistic),
        {"kind": "adj", "quart": "tp", "qf": lambda n: 1.0 / (n - 2), "recip": True},
        "RV/BV - 1; TQ / (n-2); max(1, .) with the guard 1e-300",
    )
    for kind, test in (("linear", "linear"), ("ratio", "ratio"), ("adj", "adjusted-ratio")):
        spec = {"kind": kind, "quart": "qp", "recip": kind != "linear", "sqbug": kind == "ratio"}
        add(
            f"realized-library bns_test {test}",
            rl(test),
            spec,
            "time step from the timestamps; "
            + {
                "linear": "sign as the paper's -G",
                "ratio": "RV/BV - 1 and (QP/BPV)^2 in place of QP/BV^2",
                "adjusted-ratio": "RV/BV - 1; max(1/t, .)",
            }[test],
            delta_fn=rl_delta,
            q_delta=True,
        )
    return C


def lm_paper(r: np.ndarray, K: int):
    """The paper's L_i (i >= K, 1-based: windows that fit) and T, float64."""
    a = np.abs(r)
    n = len(r)
    p = a[:-1] * a[1:]  # p[m] = |r_m||r_{m+1}|, 0-based
    from numpy.lib.stride_tricks import sliding_window_view

    w = sliding_window_view(p, K - 2)  # w[s] = p[s:s+K-2]
    # return i (0-based) uses p[i-K+1 : i-1]  -> start s = i-K+1, i = K-1 .. n-1
    sig2 = w.mean(axis=1)[: n - K + 1]
    L = a[K - 1 :] / np.sqrt(sig2)
    c = math.sqrt(2 / math.pi)
    cn = math.sqrt(2 * math.log(n)) / c - (math.log(math.pi) + math.log(math.log(n))) / (
        2 * c * math.sqrt(2 * math.log(n))
    )
    sn = 1 / (c * math.sqrt(2 * math.log(n)))
    return L, (float(L.max()) - cn) / sn, cn, sn


def lm_paper_exact_T(r: np.ndarray, K: int, L: np.ndarray, topk: int = 5):
    """T in 40 digits: the top-k float |L_i| are recomputed exactly and the largest kept."""
    import mpmath as mp

    mp.mp.dps = 40
    n = len(r)
    best = None
    for j in np.argsort(-L)[:topk]:
        i = int(j) + K - 1
        s = mp.fsum(
            abs(mp.mpf(float(r[m]))) * abs(mp.mpf(float(r[m + 1]))) for m in range(i - K + 1, i - 1)
        ) / (K - 2)
        li = abs(mp.mpf(float(r[i]))) / mp.sqrt(s)
        best = li if best is None or li > best else best
    c = mp.sqrt(2 / mp.pi)
    ln = mp.log(n)
    cn = mp.sqrt(2 * ln) / c - (mp.log(mp.pi) + mp.log(ln)) / (2 * c * mp.sqrt(2 * ln))
    sn = 1 / (c * mp.sqrt(2 * ln))
    return (best - cn) / sn


def lm_yuima_emulation(r: np.ndarray, K: int) -> float:
    """What yuima's lm.jumptest computes, from its source, in numpy.

    sigma^2_i = (pi/2) mean of the K-2 products ending at |r_{i-1}||r_i| (the current return is
    inside), the first K-3 windows extended from the first full one, i = 1 given window 1.
    """
    a = np.abs(r)
    n = len(r)
    p = a[:-1] * a[1:]
    from numpy.lib.stride_tricks import sliding_window_view

    roll = sliding_window_view(p, K - 2).mean(
        axis=1
    )  # roll[s] = mean p[s:s+K-2] = window ending m=s+K-3
    full = np.concatenate([np.full(K - 3, roll[0]), roll])  # index m = 0..n-2
    v = (math.pi / 2) * np.concatenate([[full[0]], full])
    stat = float(np.max(a / np.sqrt(v)))
    ln = math.log(n)
    cn = math.sqrt(2 * ln) - (math.log(math.pi) + math.log(ln)) / (2 * math.sqrt(2 * ln))
    sn = 1 / math.sqrt(2 * ln)
    return (stat - cn) / sn


def lm_hf_paper_form(z: np.ndarray, n: int) -> float:
    """T from highfrequency's per-return statistics L'_i = c L_i: (max|L'| - cn)/sn."""
    ln = math.log(n)
    cn = math.sqrt(2 * ln) - (math.log(math.pi) + math.log(ln)) / (2 * math.sqrt(2 * ln))
    sn = 1 / math.sqrt(2 * ln)
    return (float(np.nanmax(np.abs(z))) - cn) / sn


def synth_const(n: int, reps: int, seed: int, jumps: bool) -> np.ndarray:
    """reps days of n returns with constant volatility (1% a day), optionally the five jumps."""
    rng = np.random.default_rng(555 * n + seed)
    sd = 0.01 * math.sqrt(1.0 / n)
    r = sd * rng.standard_normal((reps, n))
    if jumps:
        idx = np.array([int(f * n) for f in (0.1, 0.3, 0.5, 0.7, 0.9)])
        r[:, idx] += np.array([8.0, -10.0, 6.0, 12.0, -7.0])[None, :] * sd
    return r


def lm_fast_T(R: np.ndarray, K: int):
    """T for the paper's window, yuima's window and highfrequency's default window (rows of R)."""
    reps, n = R.shape
    a = np.abs(R)
    p = a[:, :-1] * a[:, 1:]
    cs = np.concatenate([np.zeros((reps, 1)), np.cumsum(p, axis=1)], axis=1)  # length n
    c = math.sqrt(2 / math.pi)

    def cn_sn(m):
        ln = math.log(m)
        cn = math.sqrt(2 * ln) / c - (math.log(math.pi) + math.log(ln)) / (
            2 * c * math.sqrt(2 * ln)
        )
        return cn, 1 / (c * math.sqrt(2 * ln))

    cn, sn = cn_sn(n)
    sig2 = (cs[:, K - 2 : n - 1] - cs[:, 0 : n - K + 1]) / (K - 2)
    Tp = (np.max(a[:, K - 1 :] / np.sqrt(sig2), axis=1) - cn) / sn
    # yuima: sigma_i^2 = (pi/2)(cs[i] - cs[i-K+2])/(K-2) for i >= K-2, the first full window before
    v = np.empty((reps, n))
    v[:, K - 2 :] = (cs[:, K - 2 : n] - cs[:, 0 : n - K + 2]) / (K - 2)
    v[:, : K - 2] = v[:, [K - 2]]
    v *= math.pi / 2
    cn2 = math.sqrt(2 * math.log(n)) - (math.log(math.pi) + math.log(math.log(n))) / (
        2 * math.sqrt(2 * math.log(n))
    )
    sn2 = 1 / math.sqrt(2 * math.log(n))
    Ty = (np.max(a / np.sqrt(v), axis=1) - cn2) / sn2
    # highfrequency default: (pi/2) sum of the K-1 products of the K returns i-K+1..i, over K-2
    w = (math.pi / 2) * (cs[:, K:n] - cs[:, 1 : n - K + 1]) / (K - 2)
    ln = math.log(n + 1)
    cn3 = math.sqrt(2 * ln) - (math.log(math.pi) + math.log(ln)) / (2 * math.sqrt(2 * ln))
    sn3 = 1 / math.sqrt(2 * ln)
    Th = (np.max(a[:, K:] / np.sqrt(w), axis=1) - cn3) / sn3
    return Tp, Ty, Th


def hf_default_crit(m: int, alpha: float = 0.95) -> float:
    """highfrequency's critical value Cn + Sn (-log(-log(1 - alpha))) in its L' = c L scale."""
    ln = math.log(m)
    cn = math.sqrt(2 * ln) - (math.log(math.pi) + math.log(ln)) / (2 * math.sqrt(2 * ln))
    return cn + (-math.log(-math.log(1 - alpha))) / math.sqrt(2 * ln)


def part2(force: bool) -> None:
    import mpmath as mp

    out = CACHE / "part2.json"
    if out.exists() and not force:
        print(f"{out} exists; not recomputing (use --force)")
        return
    t0 = time.time()
    res: dict = {}
    series = build_series()
    for s in series:
        s["pr"] = _mp_prims(s["r"])
    print(f"{len(series)} series, exact primitives {time.time() - t0:.1f} s")

    # ---- BNS: every candidate against the paper with the conventions of its source
    cands = bns_candidates()
    rows = []
    for cd in cands:
        for s in series:
            n = len(s["r"])
            spec = dict(cd["spec"])
            if cd["delta_fn"] is not None:
                spec["delta"] = cd["delta_fn"](n)
                spec["q_delta"] = cd["q_delta"]
                if cd["name"].startswith("realized-library") and cd["spec"]["kind"] == "adj":
                    dt = int(round(86400e9 / n))
                    spec["one"] = 1.0 / ((n * dt) / (86400e9 - 1))  # their t^-1
            try:
                v = cd["fn"](s["r"], s["p"])
                err = None
            except Exception as e:  # noqa: BLE001
                v, err = float("nan"), f"{type(e).__name__}: {e}"[:90]
            if err is None:
                ref = bns_exact(s["pr"], spec)
                pure = bns_exact(s["pr"], {"kind": spec["kind"], "quart": spec["quart"]})
                kappa = float((s["pr"]["rv"] + s["pr"]["bv"]) / abs(s["pr"]["rv"] - s["pr"]["bv"]))
                rows.append(
                    {
                        "cand": cd["name"],
                        "group": s["group"],
                        "label": s["label"],
                        "n": n,
                        "v": v,
                        "ulps_conv": ulps(v, ref),
                        "ulps_paper": ulps(v, pure),
                        "kappa": kappa,
                        "gap_conv_per_kappa_eps": float(abs(mp.mpf(v) - ref)) / (kappa * 2.0**-52),
                        "abs_gap_conv": float(abs(mp.mpf(v) - ref)),
                        "rel_paper": float(abs(mp.mpf(v) - pure) / abs(pure)),
                        "ratio_paper": float(mp.mpf(v) / pure),
                        "note": cd["note"],
                        "err": None,
                    }
                )
            else:
                rows.append(
                    {
                        "cand": cd["name"],
                        "group": s["group"],
                        "label": s["label"],
                        "n": n,
                        "err": err,
                        "note": cd["note"],
                    }
                )
    res["bns_rows"] = rows
    print(f"BNS candidates {time.time() - t0:.1f} s")

    # ---- mfe-toolbox: what its tripower error does
    import mfe.realized.jumps as mj
    import mfe.realized.quarticity as mq
    from mfe.realized._types import RealizedResult

    orig = mq.realized_tripower_quarticity

    def fixed_tq(r):
        o = orig(r)
        return RealizedResult(value=o.value * (len(r) - 2), n_returns=o.n_returns)

    def mfe_fixed(r):
        mj.realized_tripower_quarticity = fixed_tq
        try:
            return float(mj.bns_jump_test(r).statistic)
        finally:
            mj.realized_tripower_quarticity = orig

    # on the 36 series
    eff = []
    for s in series:
        r = s["r"]
        n = len(r)
        z_ship = float(mj.bns_jump_test(r).statistic)
        z_fix = mfe_fixed(r)
        z_J = bns_np(r, "adj", "tp")
        a = np.abs(r)
        q_ratio = tpq_paper(r) / (float(np.sum(a[1:] * a[:-1])) / mu_p(1.0) ** 2) ** 2
        eff.append(
            {
                "label": s["label"],
                "group": s["group"],
                "n": n,
                "z_shipped": z_ship,
                "z_tq_fixed": z_fix,
                "z_paper_J_tp": z_J,
                "tq_over_bv2_paper": q_ratio,
            }
        )
    res["mfe_effect"] = eff

    # Monte Carlo size and power (5 planted jumps), nominal 5%
    mc = []
    for n, reps in ((78, 2000), (390, 2000), (2340, 1000)):
        for jumps in (False, True):
            R = synth_fast(n, reps, 1, jumps)
            Zs = np.array([float(mj.bns_jump_test(x).statistic) for x in R])
            Zf = np.array([mfe_fixed(x) for x in R])
            Zj = bns_np_batch(R, "adj", "tp")
            Zjr = bns_np_batch(R, "adj", "tp", recip=True)
            Zh = bns_np_batch(R, "ratio", "tp")
            Zl = bns_np_batch(R, "linear", "tp")
            Zq = bns_np_batch(R, "adj", "qp")
            two = lambda z: float(np.mean(np.abs(z) > 1.959964))  # noqa: E731
            one = lambda z: float(np.mean(z > 1.644854))  # noqa: E731
            mc.append(
                {
                    "n": n,
                    "reps": reps,
                    "jumps": jumps,
                    "mfe_shipped_two": two(Zs),
                    "mfe_shipped_one": one(Zs),
                    "mfe_tq_fixed_two": two(Zf),
                    "mfe_tq_fixed_one": one(Zf),
                    "paper_J_tp_two": two(Zj),
                    "paper_J_tp_one": one(Zj),
                    "paper_J_recip_tq_one": one(Zjr),
                    "paper_H_tp_one": one(Zh),
                    "paper_G_tp_one": one(Zl),
                    "paper_J_qp_one": one(Zq),
                    "z_mean_shipped": float(Zs.mean()),
                    "z_mean_fixed": float(Zf.mean()),
                    "z_mean_paper_J": float(Zj.mean()),
                    "z_sd_paper_J": float(Zj.std()),
                    "z_sd_shipped": float(Zs.std()),
                    "z_sd_fixed": float(Zf.std()),
                }
            )
        print(f"MC n={n} {time.time() - t0:.1f} s")
    res["mc"] = mc

    # ---- Lee and Mykland
    lm_rows = []
    ro = _r()
    fv, bl = ro.FloatVector, ro.BoolVector
    yu_lm = ro.r(
        "function(lp, K) yuima::lm.jumptest(yuima::setData(zoo::zoo(lp)), K)[[1]]$statistic"
    )
    hf_lm = ro.r("""function(p, step, by, per, K, dil, alpha) {
      idx <- as.POSIXct("2024-01-02 09:30:00", tz = "UTC") + (0:(length(p) - 1)) * step
      x <- xts::xts(p, order.by = idx); colnames(x) <- "PRICE"
      o <- highfrequency::intradayJumpTest(x, volEstimator = "RM", alignBy = by,
        alignPeriod = per, marketOpen = "09:30:00", marketClose = "16:00:00", tz = "UTC",
        RM = "rBPCov", lookBackPeriod = K, dontIncludeLast = dil, alpha = alpha)
      list(z = as.numeric(o$ztest), pr = as.numeric(o$prices$PRICE), crit = o$criticalValue)}""")
    _shim_scipy()
    import realized_library.estimators.jump_detection.lm_test as rl_lm

    grid = {
        78: ("minutes", 5, 300, [10]),
        390: ("minutes", 1, 60, [10, 30]),
        2340: ("seconds", 10, 10, [30, 100]),
        23400: ("seconds", 1, 1, [100, 300]),
    }
    beta = -math.log(-math.log(1 - 0.05))
    for s in series:
        n = len(s["r"])
        if not s["group"].startswith("synthetic") or n not in grid:
            continue
        if s["label"].endswith("seed=2"):
            continue
        by, per, step, Ks = grid[n]
        for K in Ks:
            r, p = s["r"], s["p"]
            L, T, cn, sn = lm_paper(r, K)
            Tex = lm_paper_exact_T(r, K, L)
            flagged = np.flatnonzero(cn + sn * beta < L) + K - 1
            row = {
                "label": s["label"],
                "group": s["group"],
                "n": n,
                "K": K,
                "T_paper": T,
                "T_paper_abs_gap_to_exact": float(abs(mp.mpf(T) - Tex)),
                "T_exact": float(Tex),
                "reject_paper": bool(beta < T),
                "flagged_paper": flagged.tolist(),
                "planted": s["jump_idx"].tolist(),
            }
            row["T_yuima"] = float(yu_lm(fv(np.log(p)), K)[0])
            row["T_yuima_emul"] = lm_yuima_emulation(r, K)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row["T_rl"] = float(rl_lm.compute(p, K))
                # scale (unit) dependence: the same series with returns x10
                p10 = np.exp(np.log(p[0]) + 10 * (np.log(p) - np.log(p[0])))
                row["T_rl_x10"] = float(rl_lm.compute(p10, K))
            _, T10, _, _ = lm_paper(10 * r, K)
            row["T_paper_x10"] = T10
            for dil, key in ((True, "B"), (False, "A")):
                # alpha = 0.05: the function's critical value is Cn + Sn (-log(-log(1 - alpha))),
                # which is the paper's 5% test for alpha = 0.05; its default 0.95 is not
                o = hf_lm(fv(p), step, by, per, K, bl([dil]), 0.05)
                z = np.array(o[0])
                row[f"T_hf_{key}"] = lm_hf_paper_form(z, n + 1)
                row[f"T_hf_{key}_n"] = lm_hf_paper_form(z, n)
                row[f"crit_hf_{key}"] = float(o[2][0])
                row["crit_hf_default_alpha_0.95"] = hf_default_crit(n + 1)
                row[f"flagged_hf_{key}"] = (np.flatnonzero(np.abs(z) > float(o[2][0]))).tolist()
                if key == "B":
                    c = math.sqrt(2 / math.pi)
                    zz = np.abs(z)[K:]  # 1-based j >= K+1  -> 0-based K
                    # R takes its log from libm; numpy's SIMD log differs in the last bit at a few
                    # prices, and a tiny return amplifies that, so the paper side uses libm logs
                    r_libm = np.diff(np.array([math.log(x) for x in p]))
                    pp = c * lm_paper(r_libm, K)[0][1:]  # paper L for 0-based i >= K
                    row["hf_B_perreturn_max_ulps_gap"] = float(
                        np.max(np.abs(zz - pp) / np.spacing(np.maximum(zz, pp)))
                    )
            lm_rows.append(row)
        print(f"LM {s['label']} {time.time() - t0:.1f} s")
    res["lm_rows"] = lm_rows

    # LM on the Binance data (paper, yuima, realized-library)
    lm_real = []
    for s in series:
        if not s["group"].startswith("Binance"):
            continue
        n = len(s["r"])
        for K in (30, 100):
            L, T, cn, sn = lm_paper(s["r"], K)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trl = float(rl_lm.compute(s["p"], K))
            lm_real.append(
                {
                    "label": s["label"],
                    "n": n,
                    "K": K,
                    "T_paper": T,
                    "T_yuima": float(yu_lm(fv(np.log(s["p"])), K)[0]),
                    "T_yuima_emul": lm_yuima_emulation(s["r"], K),
                    "T_rl": trl,
                    "argmax": int(np.argmax(L)) + K - 1,
                    "maxL_over_sd": float(L.max()),
                }
            )
    res["lm_real"] = lm_real

    # the cumulative-sum forms against the loop forms on the series of the table
    fc = []
    for row in lm_rows:
        if row["n"] != 2340 or row["K"] != 100 or not row["label"].endswith("seed=0"):
            continue
        r = next(x["r"] for x in series if x["label"] == row["label"])
        Tp, Ty, Th = lm_fast_T(r[None, :], row["K"])
        fc.append(
            {
                "label": row["label"],
                "d_paper": float(Tp[0] - row["T_paper"]),
                "d_yuima": float(Ty[0] - row["T_yuima_emul"]),
                "d_hf_default": float(Th[0] - row["T_hf_A"]),
            }
        )
    res["fast_check"] = fc
    print("fast vs loop forms:", fc)

    # Monte Carlo of the LM test: the paper's window, yuima's window (current return inside,
    # early windows extended) and highfrequency's default window (current return inside, K-1
    # products over K-2), all as cumulative sums; stochastic-volatility and constant-volatility days
    mc_lm = []
    for vol, n, K, reps in (
        ("sv", 390, 30, 500),
        ("sv", 2340, 100, 500),
        ("const", 2340, 200, 400),
        ("const", 23400, 1000, 200),
    ):
        for jumps in (False, True):
            R = synth_fast(n, reps, 2, jumps) if vol == "sv" else synth_const(n, reps, 2, jumps)
            Tp, Ty, Th = lm_fast_T(R, K)
            mc_lm.append(
                {
                    "vol": vol,
                    "n": n,
                    "K": K,
                    "jumps": jumps,
                    "reps": reps,
                    "reject_paper": float(np.mean(Tp > beta)),
                    "reject_yuima_window": float(np.mean(Ty > beta)),
                    "reject_hf_default_window": float(np.mean(Th > beta)),
                    "T_paper_mean": float(np.mean(Tp)),
                    "T_yuima_mean": float(np.mean(Ty)),
                    "T_hf_default_mean": float(np.mean(Th)),
                    "gumbel_mean": 0.5772156649,  # the null mean of T
                }
            )
        print(f"MC LM {vol} n={n} {time.time() - t0:.1f} s")
    res["mc_lm"] = mc_lm
    res["summary"] = {"seconds": time.time() - t0}
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"part2 done in {time.time() - t0:.1f} s -> {out}")
    print_part2(res)


def print_part2(res: dict) -> None:
    import collections

    agg: dict = collections.OrderedDict()
    for r in res["bns_rows"]:
        agg.setdefault(r["cand"], []).append(r)
    print("\n## BNS: per candidate over the 36 series")
    print(
        "| candidate | conventions | max rel gap to the paper | max abs gap to the paper "
        "with its conventions | worst gap in units of cond x 2^-52 | max cond | errors |"
    )
    for k, rs in agg.items():
        ok = [x for x in rs if x["err"] is None]
        if not ok:
            print(f"| {k} | | | | | {len(rs)} errors: {rs[0]['err']} |")
            continue
        print(
            f"| {k} | {ok[0]['note']} | {max(x['rel_paper'] for x in ok):.3e} | "
            f"{max(x['abs_gap_conv'] for x in ok):.3e} | "
            f"{max(x['gap_conv_per_kappa_eps'] for x in ok):.3g} | "
            f"{max(x['kappa'] for x in ok):.3g} | {len(rs) - len(ok)} |"
        )
    print("\n## BNS: Z of each candidate on selected series (value; ratio to the paper)")
    for lab in (
        "synthetic, no jumps n=390 seed=0",
        "synthetic, 5 jumps n=390 seed=0",
        "synthetic, 5 jumps n=23400 seed=0",
        "BTCUSDT 1m 2024-01-02",
    ):
        print(f"### {lab}")
        for k, rs in agg.items():
            for x in rs:
                if x["label"] == lab and x["err"] is None:
                    print(f"| {k} | {x['v']:.6g} | {x['ratio_paper']:.6g} |")
    print("\n## mfe-toolbox: effect of the tripower error (36 series)")
    print("| series | n | Z shipped | Z with TQ x (n-2) | Z paper J (TP) | TQ/BV^2 paper |")
    for e in res["mfe_effect"]:
        print(
            f"| {e['label']} | {e['n']} | {e['z_shipped']:.4g} | {e['z_tq_fixed']:.4g} | "
            f"{e['z_paper_J_tp']:.4g} | {e['tq_over_bv2_paper']:.4g} |"
        )
    print("\n## Monte Carlo, nominal 5%")
    for m in res["mc"]:
        print(m)
    print("\n## LM per series")
    for r in res["lm_rows"]:
        print(
            {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in r.items()
                if not k.startswith("flagged_hf") and k != "label"
            }
            | {"label": r["label"]}
        )
    print("\n## LM real")
    for r in res["lm_real"]:
        print(r)
    print("\n## LM Monte Carlo")
    for m in res["mc_lm"]:
        print(m)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=["part1", "part2"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)
    {"part1": part1, "part2": part2}[a.part](a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
