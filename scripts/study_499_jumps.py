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

* ``part3``: the online forms (one pass, decay, a hard window in rows and in clock time, zero-weight
  rows) of the bipower variation, the tripower and quadpower quarticity, the BNS ratio and the
  Lee and Mykland statistic, their state in bytes, and their gaps to the batch oracles.

* ``part4``: hfvol's Hawkes intensity against ``tick`` (needs ``numpydoc`` and ``six`` beside
  ``tick``), the online Hawkes form, and the rows each form can hold under a per-group cap.

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


# --------------------------------------------------------------------------
# Part 3: the online forms
# --------------------------------------------------------------------------
#
# A row is (clock c, return r, weight w).  A row that observes nothing (w = 0, or a return that
# is not finite) advances the clock and learns nothing: the sums decay and the window edge
# moves, no sum takes a term, and the memory of the last returns is not touched, so the row
# is as if absent.  Outputs of a row are read from the state BEFORE its update (out of sample),
# and ``weight`` is the accumulated weight before the row's update and before its own decay.
#
# Sums (a_j = |r_j|, b_j = a_j^(4/3), w the row's weight, weights decayed by lam^(clock age),
# a term takes the weight of the LATER row of its product):
#   nw = sum w            rv = sum w r^2
#   bv = sum w a_j a_{j-1}              tq = sum w b_j b_{j-1} b_{j-2}
#   qp = sum w a_j a_{j-1} a_{j-2} a_{j-3}
# Outputs: BV = bv / mu_1^2, TQ = nw tq / mu_{4/3}^3, QP = nw qp / mu_1^4, and the BNS adjusted
# ratio Z = sqrt(nw) (1 - BV/rv) / sqrt(vartheta max(1, TQ/BV^2)).  With lam = 1, no window and
# unit weights they are the paper's BV, TQ, QP and J on the rows so far.
# A hard window of `window` clock units (closed on the right: a row exactly `window` old has no
# weight) keeps the rows whose clock is above the clock of the latest snapshot at or below the
# cut; the sums are the live sums minus that snapshot decayed to now (the repo's snapshot ring).
# The window therefore holds a term for every row in it, the first one's with its predecessor
# outside the window: W returns, W products.


class JumpEngine:
    """BV, TQ, QP and the BNS ratio as one row machine (decay, hard window, zero-weight rows).

    window_kind "snapshot": the live sums minus the snapshot at the cut (the repo's ring of
    snapshots: 7 numbers a row).  window_kind "terms": a ring of each row's weighted terms
    (clock and 5 terms, 6 numbers a row) with Neumaier-compensated running sums, from which an
    expired row's term is subtracted.  compensate: the live sums are Neumaier-compensated (5 more
    numbers of state).
    """

    NSUM = 5  # nw, rv, bv, tq, qp

    def __init__(
        self,
        lam: float = 1.0,
        window: float | None = None,
        snapshot_every: float = 0.0,
        window_kind: str = "snapshot",
        compensate: bool = False,
    ):
        from collections import deque

        self.lam = float(lam)
        self.window = window
        self.every = float(snapshot_every)
        self.kind = window_kind
        self.comp = bool(compensate) or (window is not None and window_kind == "terms")
        self.s = [0.0] * self.NSUM
        self.c = [0.0] * self.NSUM  # the compensation, used when self.comp
        self.prev: list[float] = []  # |r| of the last up to 3 observed rows, newest first
        self.count = 0  # observed rows so far
        self.expired = 0  # observed rows that have left the window (terms kind)
        self.clock: float | None = None
        self.ring: deque = deque()
        self.last_snap: float | None = None
        self.win = list(self.s)
        self.cut_count = 0
        self.cut_clock = float("-inf")
        self.max_ring = 0

    # -- the row ---------------------------------------------------------
    def step(self, c: float, r: float, w: float = 1.0) -> dict:
        out = self.read()
        self.update(c, r, w)
        return out

    def _add(self, i: int, x: float) -> None:
        s = self.s[i]
        t = s + x
        if abs(s) >= abs(x):
            self.c[i] += (s - t) + x
        else:
            self.c[i] += (x - t) + s
        self.s[i] = t

    def update(self, c: float, r: float, w: float = 1.0) -> None:
        if self.clock is not None and c < self.clock:
            raise ValueError("the clock went backwards")
        if self.clock is not None and c > self.clock and self.lam < 1.0:
            f = self.lam ** (c - self.clock)
            self.s = [x * f for x in self.s]
            self.c = [x * f for x in self.c]
        self.clock = c
        observed = w > 0.0 and math.isfinite(r) and math.isfinite(w)
        if observed:
            a = abs(r)
            p = self.prev
            terms = [
                w,
                w * r * r,
                w * a * p[0] if len(p) >= 1 else 0.0,
                w * a ** (4 / 3) * p[0] ** (4 / 3) * p[1] ** (4 / 3) if len(p) >= 2 else 0.0,
                w * a * p[0] * p[1] * p[2] if len(p) >= 3 else 0.0,
            ]
            if self.comp:
                for i in range(self.NSUM):
                    self._add(i, terms[i])
            else:
                for i in range(self.NSUM):
                    self.s[i] += terms[i]
            self.prev = [a] + p[:2]
            self.count += 1
            if self.window is not None:
                if self.kind == "terms":
                    self.ring.append((c, terms))
                elif self.last_snap is None or c - self.last_snap >= self.every:
                    self.ring.append((c, self.count, *self.s))
                    self.last_snap = c
        if self.window is None:
            self.win = [x + y for x, y in zip(self.s, self.c, strict=True)]
            return
        cutoff = c - self.window
        ring = self.ring
        if self.kind == "terms":
            while ring and ring[0][0] <= cutoff:
                ck, tm = ring.popleft()
                f = self.lam ** (c - ck) if self.lam < 1.0 else 1.0
                for i in range(self.NSUM):
                    self._add(i, -f * tm[i])
                self.expired += 1
            self.win = [x + y for x, y in zip(self.s, self.c, strict=True)]
            self.cut_count, self.cut_clock = self.expired, cutoff
            self.max_ring = max(self.max_ring, len(ring))
            return
        while len(ring) >= 2 and ring[1][0] <= cutoff:
            ring.popleft()
        if ring and ring[0][0] <= cutoff:
            snap = ring[0]
            f = self.lam ** (c - snap[0]) if self.lam < 1.0 else 1.0
            self.win = [x - f * y for x, y in zip(self.s, snap[2:], strict=True)]
            self.cut_clock, self.cut_count = snap[0], snap[1]
        else:
            self.win = list(self.s)
            self.cut_clock, self.cut_count = float("-inf"), 0
        self.max_ring = max(self.max_ring, len(ring))

    # -- the outputs, from the state as it stands -----------------------
    def read(self) -> dict:
        nw, rv, bv, tq, qp = self.win
        nan = float("nan")
        bvv = bv * (math.pi / 2)
        out = {"weight": nw, "rv": rv, "bv": bvv if bv > 0 else nan}
        out["tpq"] = nw * tq / mu_p(4 / 3) ** 3 if tq > 0 else nan
        out["qpq"] = nw * qp * (math.pi / 2) ** 2 if qp > 0 else nan
        if nw > 0 and rv > 0 and bv > 0 and tq > 0:
            avar = max(1.0, out["tpq"] / bvv**2)
            out["z_bns"] = math.sqrt(nw) * (1 - bvv / rv) / math.sqrt(THETA * avar)
        else:
            out["z_bns"] = nan
        return out

    # -- the state ---------------------------------------------------------
    def state_bytes(self) -> int:
        """Bytes of the numbers held: 8 each (f64 sums and |r|, i64 clock and counts)."""
        fixed = self.NSUM * 8 + 3 * 8 + 1 + 8 + 8 + 8  # sums, |r|s, n prev, clock, last snap, count
        if self.comp:
            fixed += self.NSUM * 8 + 8  # the compensations and the expired count
        per = (1 + self.NSUM) * 8 if self.kind == "terms" else (2 + self.NSUM) * 8
        return fixed + len(self.ring) * per


# -- the sliding sum of the Lee and Mykland window, four ways -----------------------------------


class _WinNaive:
    """Ring of the last cap terms and a running sum that adds the new term and subtracts the old."""

    def __init__(self, cap):
        from collections import deque

        self.cap, self.q, self.s = cap, deque(), 0.0

    def push(self, x):
        self.q.append(x)
        self.s += x
        if len(self.q) > self.cap:
            self.s -= self.q.popleft()

    def total(self):
        return self.s if len(self.q) == self.cap else float("nan")

    def nbytes(self):
        return 8 * (self.cap + 1)


class _WinNeumaier(_WinNaive):
    """The same with Neumaier compensation on the addition and on the subtraction."""

    def __init__(self, cap):
        super().__init__(cap)
        self.c = 0.0

    def _add(self, x):
        t = self.s + x
        if abs(self.s) >= abs(x):
            self.c += (self.s - t) + x
        else:
            self.c += (x - t) + self.s
        self.s = t

    def push(self, x):
        self.q.append(x)
        self._add(x)
        if len(self.q) > self.cap:
            self._add(-self.q.popleft())

    def total(self):
        return self.s + self.c if len(self.q) == self.cap else float("nan")

    def nbytes(self):
        return 8 * (self.cap + 2)


class _WinTwoStack:
    """Two stacks: pushes go on the back (running sum), pops come off a front of suffix sums."""

    def __init__(self, cap):
        self.cap, self.back, self.bsum, self.front, self.n = (
            cap,
            [],
            0.0,
            [],
            0,
        )  # front: (x, suffix)
        self.peak = 0

    def push(self, x):
        self.back.append(x)
        self.bsum += x
        self.n += 1
        if self.n > self.cap:
            if not self.front:  # flip: front holds the suffix sums of the back, oldest on top
                acc = 0.0
                for v in reversed(self.back):
                    acc += v
                    self.front.append((v, acc))
                self.back, self.bsum = [], 0.0
            self.front.pop()
            self.n -= 1
        self.peak = max(self.peak, len(self.back) + 2 * len(self.front))

    def total(self):
        if self.n < self.cap:
            return float("nan")
        return (self.front[-1][1] if self.front else 0.0) + self.bsum

    def nbytes(self):
        return 8 * (
            2 * self.cap + 1
        )  # worst case: every term once in the back or twice in the front


class _WinSnapshot:
    """Cumulative sums, one per row, and the window as the difference of two of them."""

    def __init__(self, cap):
        from collections import deque

        self.cap, self.cum, self.n = cap, deque([0.0]), 0
        self.s = 0.0

    def push(self, x):
        self.s += x
        self.cum.append(self.s)
        self.n += 1
        if len(self.cum) > self.cap + 1:
            self.cum.popleft()

    def total(self):
        return self.s - self.cum[0] if self.n >= self.cap else float("nan")

    def nbytes(self):
        return 8 * (self.cap + 2)


WINDOW_SUMS = {
    "naive": _WinNaive,
    "neumaier": _WinNeumaier,
    "two-stack": _WinTwoStack,
    "snapshot": _WinSnapshot,
}


class LeeMyklandEngine:
    """The Lee and Mykland statistic as a row machine.

    sigma^2 of a row is read from the rows before it: the mean of the products |r_j||r_{j-1}| of
      rows    the last K-2 products (the paper's window), a ring and one of four running sums
      decay   an exponentially weighted mean, sum(lam^age w p) / sum(lam^age w)
      clock   the products of the rows within `window` clock units: a ring of (clock, p, w) with
              Neumaier-compensated sums, an expired row subtracted as it left
    and L = |r| / sigma; the test statistic T = (max |L| - C_n) / S_n over the rows tested so far
    (n = the count, or n_ref where it is fixed), and a row is flagged when |L| > C_n + S_n beta*.
    """

    def __init__(
        self,
        mode: str,
        K: int | None = None,
        sum_kind: str = "two-stack",
        lam: float = 1.0,
        window: float | None = None,
        n_ref: int | None = None,
        alpha: float = 0.05,
    ):
        from collections import deque

        self.mode, self.K, self.lam, self.window, self.n_ref = mode, K, lam, window, n_ref
        self.beta = -math.log(-math.log(1 - alpha))
        self.prev: float | None = None  # |r| of the last observed row
        self.clock: float | None = None
        self.sum = WINDOW_SUMS[sum_kind](K - 2) if mode == "rows" else None
        self.sp = [0.0, 0.0]  # decayed sum of products and of weights, and their compensations
        self.sw = [0.0, 0.0]
        self.ring: deque = deque()  # clock mode: (clock, p, w)
        self.count = 0
        self.tests = 0
        self.maxabs = 0.0
        self.max_ring = 0

    @staticmethod
    def _nadd(acc, x):
        s = acc[0]
        t = s + x
        if abs(s) >= abs(x):
            acc[1] += (s - t) + x
        else:
            acc[1] += (x - t) + s
        acc[0] = t

    def step(self, c: float, r: float, w: float = 1.0) -> dict:
        out = {"L": float("nan"), "T": float("nan"), "flag": False}
        if self.clock is not None and c < self.clock:
            raise ValueError("the clock went backwards")
        if self.mode != "rows" and self.clock is not None and c > self.clock and self.lam < 1.0:
            f = self.lam ** (c - self.clock)
            self.sp = [x * f for x in self.sp]
            self.sw = [x * f for x in self.sw]
        self.clock = c
        if self.mode == "clock":
            self._expire(c)
        observed = w > 0.0 and math.isfinite(r) and math.isfinite(w)
        if not observed:
            return out
        a = abs(r)
        sig2 = self._sigma2()
        if sig2 == sig2 and sig2 > 0:  # a test: the row has a volatility estimate
            L = a / math.sqrt(sig2)
            self.tests += 1
            self.maxabs = max(self.maxabs, L)
            n = self.n_ref or self.tests
            out["L"] = L
            if n >= 3:  # C_n and S_n need log log n
                cn, sn = lm_cn_sn(n)
                out.update(T=(self.maxabs - cn) / sn, flag=cn + sn * self.beta < L)
        if self.prev is not None:
            p = a * self.prev
            if self.mode == "rows":
                self.sum.push(p)
            else:
                self._nadd(self.sp, w * p)
                self._nadd(self.sw, w)
                if self.mode == "clock":
                    self.ring.append((c, w * p, w))
                    self.max_ring = max(self.max_ring, len(self.ring))
        self.prev = a
        self.count += 1
        return out

    def _expire(self, c):
        cutoff = c - self.window
        ring = self.ring
        while ring and ring[0][0] <= cutoff:
            ck, wp, w = ring.popleft()
            f = self.lam ** (c - ck) if self.lam < 1.0 else 1.0
            self._nadd(self.sp, -f * wp)
            self._nadd(self.sw, -f * w)

    def _sigma2(self):
        if self.mode == "rows":
            return self.sum.total() / (self.K - 2)
        sw = self.sw[0] + self.sw[1]
        return (self.sp[0] + self.sp[1]) / sw if sw > 0 else float("nan")

    def state_bytes(self) -> int:
        base = 8 + 8 + 8 + 8 + 8 + 8  # prev, clock, count, tests, max |L|, spare
        if self.mode == "rows":
            return base + self.sum.nbytes()
        if self.mode == "decay":
            return base + 32  # sp, sw and their compensations
        return base + 32 + len(self.ring) * 24


def lm_cn_sn(n: int):
    c = math.sqrt(2 / math.pi)
    ln = math.log(n)
    cn = math.sqrt(2 * ln) / c - (math.log(math.pi) + math.log(ln)) / (2 * c * math.sqrt(2 * ln))
    return cn, 1 / (c * math.sqrt(2 * ln))


def state_table() -> list[dict]:
    """State in bytes of the online forms: measured on the engines, and the minimal formulas."""
    rows = []

    def measured(name, make, drive):
        e1, e2 = make(10), make(100)
        drive(e1, 10)
        drive(e2, 100)
        n1, n2 = (
            len(e1.ring) if hasattr(e1, "ring") else 0,
            len(e2.ring) if hasattr(e2, "ring") else 0,
        )
        if n2 != n1:
            per = (e2.state_bytes() - e1.state_bytes()) / (n2 - n1)
            fixed = e1.state_bytes() - per * n1
        else:
            per, fixed = 0.0, float(e1.state_bytes())
        rows.append({"form": name, "fixed_bytes": fixed, "per_row_bytes": per})

    rng = np.random.default_rng(1)
    x = (rng.standard_normal(2000) * 1e-3).tolist()

    def drive_j(e, W):
        for i in range(400):
            e.update(float(i), x[i], 1.0)

    def drive_lm(e, K):
        for i in range(400):
            e.step(float(i), x[i], 1.0)

    measured("BNS engine, one pass / decay", lambda W: JumpEngine(), drive_j)
    measured("BNS engine, compensated sums", lambda W: JumpEngine(compensate=True), drive_j)
    measured(
        "BNS engine, hard window, snapshot ring",
        lambda W: JumpEngine(window=float(W), window_kind="snapshot"),
        drive_j,
    )
    measured(
        "BNS engine, hard window, term ring (compensated)",
        lambda W: JumpEngine(window=float(W), window_kind="terms"),
        drive_j,
    )
    measured("Lee-Mykland, decay", lambda K: LeeMyklandEngine("decay", lam=0.99), drive_lm)
    measured(
        "Lee-Mykland, clock window, term ring (compensated)",
        lambda W: LeeMyklandEngine("clock", window=float(W)),
        drive_lm,
    )
    for kind in WINDOW_SUMS:
        e1, e2 = (
            LeeMyklandEngine("rows", K=10, sum_kind=kind),
            LeeMyklandEngine("rows", K=110, sum_kind=kind),
        )
        per = (e2.state_bytes() - e1.state_bytes()) / 100
        rows.append(
            {
                "form": f"Lee-Mykland, window of K rows, {kind} sum",
                "fixed_bytes": e1.state_bytes() - per * 10,
                "per_row_bytes": per,
            }
        )
    return rows


def minimal_table() -> list[dict]:
    """The numbers each output needs, at 8 bytes each (formulas, not measured)."""
    f = 8
    return [
        {
            "output": "bipower variation",
            "constant": (1 + 1 + 2) * f,
            "ring_per_row_in_window": 2 * f,
            "numbers": "1 sum, last |r|, clock, count; ring: clock + 1 term",
        },
        {
            "output": "tripower quarticity",
            "constant": (2 + 2 + 2) * f,
            "ring_per_row_in_window": 3 * f,
            "numbers": "weight and 1 sum, last 2 |r|, clock, count; ring: clock + 2 terms",
        },
        {
            "output": "BNS ratio, TQ",
            "constant": (4 + 2 + 2) * f,
            "ring_per_row_in_window": 5 * f,
            "numbers": "4 sums, last 2 |r|, clock, count; ring: clock + 4 terms",
        },
        {
            "output": "BNS ratio, QP",
            "constant": (4 + 3 + 2) * f,
            "ring_per_row_in_window": 5 * f,
            "numbers": "4 sums, last 3 |r|, clock, count; ring: clock + 4 terms",
        },
        {
            "output": "Lee-Mykland, K rows",
            "constant": 6 * f,
            "ring_per_row_in_window": f,
            "numbers": "prev, clock, count, tests, max, 1 sum; ring: one product per window row",
        },
    ]


def _obs(clocks, r, weights):
    """The observed rows as arrays (clock, |r|, w, r)."""
    r = np.asarray(r, dtype=float)
    return (np.asarray(clocks, dtype=float), np.abs(r), np.asarray(weights, dtype=float), r)


def _exact_sums(obs, lam, T, after_count):
    """The sums of the observed rows after_count..T at the clock of row T, as mpmath numbers.

    The reference for the decayed and windowed forms: each term in float64 (the weight
    lam**age by pow, the product of the |r|), summed exactly with fsum.  Its own error is the
    rounding of the terms, about one ulp and no more; what it leaves out is the accumulation
    of the running sums, which is what the gaps measure.  The terms use the observed
    predecessors wherever they are, so the window's first row keeps its product.
    """
    import mpmath as mp

    c, a, w, r = obs
    j = np.arange(after_count, T + 1)
    f = lam ** (c[T] - c[j]) if lam < 1.0 else np.ones(len(j))
    wa = w[j] * f
    b = a ** (4 / 3)
    jm = [np.maximum(j - i, 0) for i in range(4)]
    t2 = np.where(j >= 1, a[j] * a[jm[1]], 0.0)
    t3 = np.where(j >= 2, b[j] * b[jm[1]] * b[jm[2]], 0.0)
    t4 = np.where(j >= 3, a[j] * a[jm[1]] * a[jm[2]] * a[jm[3]], 0.0)
    mp.mp.dps = 40
    return tuple(mp.mpf(math.fsum(x)) for x in (wa, wa * r[j] ** 2, wa * t2, wa * t3, wa * t4))


def _exact_outputs(sums):
    import mpmath as mp

    nw, rv, bv, tq, qp = sums
    mu1sq = 2 / mp.pi
    mu43 = mp.mpf(2) ** (mp.mpf(2) / 3) * mp.gamma(mp.mpf(7) / 6) / mp.gamma(mp.mpf(1) / 2)
    BV = bv / mu1sq
    TQ = nw * tq / mu43**3
    QP = nw * qp / mu1sq**2
    theta = mp.pi**2 / 4 + mp.pi - 5
    Z = mp.sqrt(nw) * (1 - BV / rv) / mp.sqrt(theta * max(mp.mpf(1), TQ / BV**2))
    return {"weight": nw, "rv": rv, "bv": BV, "tpq": TQ, "qpq": QP, "z_bns": Z}


def _ulp_gap(v, ref) -> float:
    import mpmath as mp

    if v != v:
        return float("inf")
    f = float(ref)
    return float(abs(mp.mpf(v) - ref) / mp.mpf(math.ulp(f))) if f != 0 else float(abs(v))


def _cond_gap(v, ref, kappa) -> float:
    """|v - ref| in units of kappa 2^-52 (a difference is amplified by kappa)."""
    import mpmath as mp

    return float(abs(mp.mpf(v) - ref)) / (kappa * 2.0**-52) if v == v else float("inf")


def _kappa(sums) -> float:
    import mpmath as mp

    return float((sums[1] + sums[2] * mp.pi / 2) / abs(sums[1] - sums[2] * mp.pi / 2))


def _row_gaps(eng, obs, lam, i, gaps, cut=None):
    """Update gaps (max) with the engine's outputs at row i against the exact ones."""
    sums = _exact_sums(obs, lam, i, eng.cut_count if cut is None else cut)
    ex = _exact_outputs(sums)
    got = eng.read()
    for k in gaps:
        g = _ulp_gap(got[k], ex[k]) if k != "z_bns" else _cond_gap(got[k], ex[k], _kappa(sums))
        gaps[k] = max(gaps[k], g)


def _samples(n, lo, count=25):
    return set(range(lo, n, max(1, (n - lo) // count))) | {n - 1}


def part3(force: bool) -> None:
    import itertools

    out = CACHE / "part3.json"
    if out.exists() and not force:
        print(f"{out} exists; not recomputing (use --force)")
        return
    t0 = time.time()
    res: dict = {"state_measured": state_table(), "state_minimal": minimal_table()}
    series = {s["label"]: s for s in build_series()}
    pick = {
        "synthetic, 5 jumps n=23400 seed=0": "J23400",
        "synthetic, no jumps n=2340 seed=0": "N2340",
        "BTCUSDT 1m 2024-01-05": "BTC",
        "ETCUSDT trades 10 s 2024-01-02": "ETC10s",
    }
    S = {tag: series[lab] for lab, tag in pick.items()}
    fields = ("bv", "tpq", "qpq", "rv", "z_bns", "weight")

    # ---- one pass, every sampled prefix against the exact paper values
    onepass = []
    for comp, (tag, s) in itertools.product((False, True), S.items()):
        r, n = s["r"], len(s["r"])
        eng = JumpEngine(compensate=comp)
        obs = _obs(np.arange(n, dtype=float), r, np.ones(n))
        gaps = dict.fromkeys(fields, 0.0)
        smp = _samples(n, 10)
        for i in range(n):
            eng.update(float(i), float(r[i]), 1.0)
            if i in smp:
                _row_gaps(eng, obs, 1.0, i, gaps, cut=0)
        onepass.append(
            {
                "series": tag,
                "n": n,
                "compensated": comp,
                "max_gap": gaps,
                "state_bytes": eng.state_bytes(),
                "final": eng.read(),
            }
        )
    res["onepass"] = onepass
    print(f"one pass {time.time() - t0:.1f} s")

    # ---- one pass against the oracles of parts 1 and 2 on the whole series
    _shim_scipy()
    import hfvol

    ro = _r()
    fv = ro.FloatVector
    rbp, rtp = ro.r["rBPCov"], ro.r["rTPQuar"]
    yu = ro.r(
        "function(lp, r, type, adj) yuima::bns.test(yuima::setData(zoo::zoo(lp)), r = r,"
        " type = type, adj = adj)[[1]]$statistic"
    )
    hfb = ro.r(
        "function(x, type, mx, iq) highfrequency::BNSjumpTest(x, type = type, max = mx,"
        " IQestimator = iq)$ztest"
    )
    vs = []
    for comp, (tag, s) in itertools.product((False, True), S.items()):
        r, p = s["r"], s["p"]
        n = len(r)
        eng = JumpEngine(compensate=comp)
        for i in range(n):
            eng.update(float(i), float(r[i]), 1.0)
        o = eng.read()
        bv_hfvol = hfvol.bipower_variation(r)
        bv_r = float(rbp(fv(r))[0])
        tq_r = float(rtp(fv(r))[0]) * (n - 2) / n  # highfrequency's n/(n-2) divided out
        z_yu = float(yu(fv(np.log(p)), fv([4 / 3] * 3), "ratio", ro.BoolVector([True]))[0])
        z_yu *= math.sqrt(n / (n + 1))  # yuima's sqrt(n+1) prices divided out
        z_hf = float(hfb(fv(r), "ratio", ro.BoolVector([True]), "TP")[0])
        nw, rv, bvs, tqs, _ = eng.win
        bvv = bvs * math.pi / 2
        tqv = o["tpq"] * n / (n - 2)  # highfrequency's tripower carries n/(n-2)
        z_hf_online = math.sqrt(n) * (1 - bvv / rv) / math.sqrt(THETA * max(1.0, tqv / bvv**2))
        vs.append(
            {
                "series": tag,
                "n": n,
                "compensated": comp,
                "bv_vs_hfvol_ulps": abs(o["bv"] - bv_hfvol) / math.ulp(bv_hfvol),
                "bv_vs_rBPCov_ulps": abs(o["bv"] - bv_r) / math.ulp(bv_r),
                "tpq_vs_rTPQuar_over_factor_ulps": abs(o["tpq"] - tq_r) / math.ulp(tq_r),
                "z_vs_yuima_TP_rel": abs(o["z_bns"] - z_yu) / abs(z_yu),
                "z_vs_highfrequency_J_rel": abs(z_hf_online - z_hf) / abs(z_hf),
            }
        )
    res["vs_oracles"] = vs
    print(f"oracles {time.time() - t0:.1f} s")

    # ---- decay: lam per clock unit, row clock and an irregular clock, weights not all 1
    decay = []
    for comp, tag in itertools.product((False, True), ("J23400", "BTC")):
        r, n = S[tag]["r"], len(S[tag]["r"])
        g = np.random.default_rng(7)
        clocks_irr = np.cumsum(g.exponential(1.0, n))
        for kind, clocks in (
            ("row clock", np.arange(n, dtype=float)),
            ("irregular clock", clocks_irr),
        ):
            for hl in (50.0, 500.0):
                lam = 0.5 ** (1.0 / hl)
                weights = 0.5 + g.random(n)
                eng = JumpEngine(lam=lam, compensate=comp)
                obs = _obs(clocks, r, weights)
                gaps = dict.fromkeys(fields, 0.0)
                smp = _samples(n, 10)
                for i in range(n):
                    eng.update(float(clocks[i]), float(r[i]), float(weights[i]))
                    if i in smp:
                        _row_gaps(eng, obs, lam, i, gaps, cut=0)
                decay.append(
                    {
                        "series": tag,
                        "clock": kind,
                        "half_life": hl,
                        "compensated": comp,
                        "n": n,
                        "max_gap": gaps,
                        "state_bytes": eng.state_bytes(),
                    }
                )
    res["decay"] = decay
    print(f"decay {time.time() - t0:.1f} s")

    # ---- hard window, in rows and in clock time, with and without a decay
    windows = []
    for tag in ("J23400", "BTC"):
        r, n = S[tag]["r"], len(S[tag]["r"])
        g = np.random.default_rng(11)
        clocks_irr = np.cumsum(g.exponential(1.0, n))
        rowc = np.arange(n, dtype=float)
        for kind, clocks, wk, every_div in (
            ("rows", rowc, "snapshot", 0),
            ("rows", rowc, "terms", 0),
            ("clock", clocks_irr, "snapshot", 0),
            ("clock", clocks_irr, "terms", 0),
            ("clock, snapshot every window/20", clocks_irr, "snapshot", 20),
        ):
            for W in (100.0, 1000.0):
                for lam in (1.0, 0.5 ** (1.0 / (W / 4))):
                    every = W / every_div if every_div else 0.0
                    eng = JumpEngine(lam=lam, window=W, snapshot_every=every, window_kind=wk)
                    obs = _obs(clocks, r, np.ones(n))
                    gaps = dict.fromkeys(fields, 0.0)
                    smp = _samples(n, int(W) + 5)
                    rows_in = 0
                    edge_ok = True
                    for i in range(n):
                        eng.update(float(clocks[i]), float(r[i]), 1.0)
                        if i in smp:
                            _row_gaps(eng, obs, lam, i, gaps)
                            rows_in = max(rows_in, i + 1 - eng.cut_count)
                        if every == 0.0 and i % 331 == 0:
                            edge_ok &= (i + 1 - eng.cut_count) == int(
                                np.sum(clocks[: i + 1] > clocks[i] - W)
                            )
                    per = 7 * 8 if wk == "snapshot" else 6 * 8
                    windows.append(
                        {
                            "series": tag,
                            "kind": kind,
                            "window_kind": wk,
                            "W": W,
                            "lam": lam,
                            "n": n,
                            "max_gap": gaps,
                            "max_ring": eng.max_ring,
                            "max_rows_in_window_sampled": rows_in,
                            "bytes_at_max_ring": JumpEngine(window=W, window_kind=wk).state_bytes()
                            + eng.max_ring * per,
                            "edge_is_exact": edge_ok if every == 0.0 else None,
                        }
                    )
    res["windows"] = windows
    print(f"windows {time.time() - t0:.1f} s")

    # ---- the windowed forms against the batch oracles (a window of W rows, no decay)
    batch = []
    for tag, wk in itertools.product(("J23400", "BTC"), ("snapshot", "terms")):
        r, n = S[tag]["r"], len(S[tag]["r"])
        for W in (100, 1000):
            eng = JumpEngine(window=float(W), window_kind=wk)
            worst = [0.0, 0.0, 0.0]
            for i in range(n):
                eng.update(float(i), float(r[i]), 1.0)
                if i >= W + 2 and i % 997 == 0:
                    o = eng.read()
                    seg = r[i - W : i + 1]  # W + 1 returns, W products
                    bvh, bvr = hfvol.bipower_variation(seg), float(rbp(fv(seg))[0])
                    seg3 = r[i - W - 1 : i + 1]  # W + 2 returns, W triple products
                    N = len(seg3)
                    tqh = float(rtp(fv(seg3))[0]) * W**2 / N**2  # N N/(N-2) -> W
                    worst[0] = max(worst[0], abs(o["bv"] - bvh) / math.ulp(bvh))
                    worst[1] = max(worst[1], abs(o["bv"] - bvr) / math.ulp(bvr))
                    worst[2] = max(worst[2], abs(o["tpq"] - tqh) / math.ulp(tqh))
            batch.append(
                {
                    "series": tag,
                    "window_kind": wk,
                    "W": W,
                    "bv_vs_hfvol_ulps": worst[0],
                    "bv_vs_rBPCov_ulps": worst[1],
                    "tpq_vs_rTPQuar_scaled_ulps": worst[2],
                }
            )
    res["windows_vs_batch"] = batch
    print(f"batch {time.time() - t0:.1f} s")

    # ---- zero-weight rows learn nothing
    zero = []
    garbage = [float("nan"), float("inf"), 1e300, -1e300, 0.37, -1e-300]
    for tag in ("N2340", "BTC"):
        r, n = S[tag]["r"], len(S[tag]["r"])
        zrng = np.random.default_rng(5)
        for lam, W, wk, mode, shift in (
            (
                1.0,
                None,
                "snapshot",
                "lam = 1, no window; zero-weight row at its neighbour's clock",
                0.0,
            ),
            (
                0.5 ** (1 / 100),
                None,
                "snapshot",
                "decay; zero-weight row half a clock unit later",
                0.5,
            ),
            (
                1.0,
                200.0,
                "snapshot",
                "window of 200 clock units, snapshot ring; zero-weight row half a unit later",
                0.5,
            ),
            (
                1.0,
                200.0,
                "terms",
                "window of 200 clock units, term ring; zero-weight row half a unit later",
                0.5,
            ),
        ):
            ext = set(zrng.choice(n - 1, n // 3, replace=False).tolist())
            c = 0.0
            stream, oc, orr = [], [], []
            for i in range(n):
                c += 1.0
                stream.append((c, float(r[i]), 1.0))
                oc.append(c)
                orr.append(float(r[i]))
                if i in ext:
                    stream.append((c + shift, garbage[int(zrng.integers(0, len(garbage)))], 0.0))
                    c += shift
            eng_a = JumpEngine(lam=lam, window=W, window_kind=wk)
            eng_b = JumpEngine(lam=lam, window=W, window_kind=wk)
            poison = False
            for cc, rr, ww in stream:
                eng_a.update(cc, rr, ww)
                poison |= not all(math.isfinite(x) for x in eng_a.s)
            for cc, rr in zip(oc, orr, strict=True):
                eng_b.update(cc, rr, 1.0)
            oa, ob = eng_a.read(), eng_b.read()
            bitwise = all(oa[k] == ob[k] for k in oa)
            gap_ab = max(abs(oa[k] - ob[k]) / math.ulp(ob[k]) for k in oa if k != "z_bns")
            obs = _obs(oc, orr, np.ones(len(oc)))
            gaps = dict.fromkeys(fields, 0.0)
            _row_gaps(eng_a, obs, lam, len(oc) - 1, gaps)
            zero.append(
                {
                    "series": tag,
                    "mode": mode,
                    "n_obs": len(oc),
                    "n_zero_weight": len(stream) - len(oc),
                    "bitwise_equal_to_stream_without": bool(bitwise),
                    "max_ulps_to_stream_without": gap_ab,
                    "max_ulps_to_exact": max(v for k, v in gaps.items() if k != "z_bns"),
                    "state_has_nan_or_inf": bool(poison),
                }
            )
    e = JumpEngine(lam=0.99)
    for i in range(5):
        e.update(float(i), float("nan") if i % 2 else 3.0, 0.0)
    for i in range(5, 40):
        e.update(float(i), 1e-3 * (1 + (i % 3)), 1.0)
    zero.append(
        {
            "series": "the first 5 rows have weight 0",
            "mode": "decay",
            "n_obs": 35,
            "n_zero_weight": 5,
            "state_has_nan_or_inf": not all(math.isfinite(x) for x in e.s),
            "output_finite": all(math.isfinite(v) for v in e.read().values()),
        }
    )
    res["zero_weight"] = zero
    print(f"zero weight {time.time() - t0:.1f} s")

    # ---- Lee and Mykland: the window of K rows, four running sums
    lm = []
    for tag, Ks in (("J23400", (100, 300)), ("N2340", (30, 100)), ("BTC", (30, 100))):
        r, n = S[tag]["r"], len(S[tag]["r"])
        for K in Ks:
            L, T, cn, sn = lm_paper(r, K)
            beta = -math.log(-math.log(1 - 0.05))
            flags_paper = (np.flatnonzero(cn + sn * beta < L) + K - 1).tolist()
            for kind in WINDOW_SUMS:
                eng = LeeMyklandEngine("rows", K=K, sum_kind=kind, n_ref=n)
                Ls = np.full(n, np.nan)
                flags = []
                for i in range(n):
                    o = eng.step(float(i), float(r[i]), 1.0)
                    Ls[i] = o["L"]
                    if o["flag"]:
                        flags.append(i)
                lm.append(
                    {
                        "series": tag,
                        "n": n,
                        "K": K,
                        "sum": kind,
                        "L_vs_paper_float_max_ulps": float(
                            np.nanmax(np.abs(Ls[K - 1 :] - L) / np.spacing(L))
                        ),
                        "T_online": float((np.nanmax(Ls) - cn) / sn),
                        "T_paper": T,
                        "flags_equal_paper": flags == flags_paper,
                        "n_flags": len(flags),
                        "state_bytes": eng.state_bytes(),
                        "peak_terms_two_stack": eng.sum.peak if kind == "two-stack" else None,
                    }
                )
    res["lm_rows"] = lm

    # the four sums when an outlier leaves the window (relative error of the window sum)
    cancel = []
    base = S["N2340"]["r"].copy()
    K = 100
    for amp in (1.0, 1e3, 1e6, 1e9):
        r = base.copy()
        r[500] = amp * float(np.std(base))
        a = np.abs(r)
        pr = a[1:] * a[:-1]
        wnd = {k: WINDOW_SUMS[k](K - 2) for k in WINDOW_SUMS}
        worst = dict.fromkeys(WINDOW_SUMS, 0.0)
        for i in range(len(pr)):
            for k in wnd:
                wnd[k].push(float(pr[i]))
            if i >= 700 and i % 17 == 0:  # after the outlier has left the window
                ex = math.fsum(pr[i - (K - 3) : i + 1])
                for k in wnd:
                    worst[k] = max(worst[k], abs(wnd[k].total() - ex) / ex)
        cancel.append({"outlier_sd": amp, **{f"rel_err_{k}": v for k, v in worst.items()}})
    res["window_sum_outlier"] = cancel

    # decay and clock-time forms of Lee and Mykland against the exact weighted mean
    import mpmath as mp

    lm2 = []
    for tag in ("J23400", "BTC"):
        r, n = S[tag]["r"], len(S[tag]["r"])
        g = np.random.default_rng(3)
        clocks_irr = np.cumsum(g.exponential(1.0, n))
        for mode, kw in (("decay", {"lam": 0.5 ** (1 / 100)}), ("clock", {"window": 100.0})):
            eng = LeeMyklandEngine(mode, n_ref=n, **kw)
            lam = kw.get("lam", 1.0)
            a = np.abs(r)
            worst = 0.0
            for i in range(n):
                o = eng.step(float(clocks_irr[i]), float(r[i]), 1.0)
                if i > 300 and i % 733 == 0 and o["L"] == o["L"]:
                    mp.mp.dps = 40
                    cut = float(clocks_irr[i]) - kw.get("window", math.inf)
                    num = den = mp.mpf(0)
                    for j in range(1, i):
                        if mode == "clock" and clocks_irr[j] <= cut:
                            continue
                        f = (
                            mp.power(
                                mp.mpf(lam),
                                mp.mpf(float(clocks_irr[i])) - mp.mpf(float(clocks_irr[j])),
                            )
                            if mode == "decay"
                            else 1
                        )
                        num += f * mp.mpf(float(a[j])) * mp.mpf(float(a[j - 1]))
                        den += f
                    Lex = mp.mpf(float(a[i])) / mp.sqrt(num / den)
                    worst = max(worst, float(abs(mp.mpf(o["L"]) - Lex) / (Lex * mp.mpf(2.0**-52))))
            lm2.append(
                {
                    "series": tag,
                    "mode": mode,
                    "max_rel_gap_in_eps": worst,
                    "max_ring": eng.max_ring,
                    "state_bytes": eng.state_bytes(),
                }
            )
    res["lm_other"] = lm2
    print(f"LM {time.time() - t0:.1f} s")

    # ---- the state is bounded: the ring and the bytes stay put over many windows of rows
    bound = []
    r = S["J23400"]["r"]
    for wk, W in itertools.product(("snapshot", "terms"), (100, 1000)):
        eng = JumpEngine(window=float(W), window_kind=wk)
        sizes = []
        for i in range(len(r)):
            eng.update(float(i), float(r[i]), 1.0)
            if i % 1000 == 0:
                sizes.append(eng.state_bytes())
        per = 7 * 8 if wk == "snapshot" else 6 * 8
        bound.append(
            {
                "window_kind": wk,
                "W": W,
                "rows": len(r),
                "max_ring": eng.max_ring,
                "bytes_over_checkpoints": [min(sizes), max(sizes)],
                "formula_bytes": JumpEngine(window=float(W), window_kind=wk).state_bytes()
                + (W + (1 if wk == "snapshot" else 0)) * per,
            }
        )
    res["bounded"] = bound

    # ---- chunks and resume: saving the state at a chunk edge and loading it changes no bit
    import copy
    import pickle

    resume = []
    r = S["N2340"]["r"]
    n = len(r)
    clocks = np.cumsum(np.random.default_rng(2).exponential(1.0, n))
    makers = {
        "BNS one pass": lambda: JumpEngine(),
        "BNS decay": lambda: JumpEngine(lam=0.99),
        "BNS window, snapshot ring": lambda: JumpEngine(window=100.0),
        "BNS window, term ring": lambda: JumpEngine(window=100.0, window_kind="terms"),
        "BNS decay and window": lambda: JumpEngine(lam=0.99, window=100.0, window_kind="terms"),
        "LM K rows, Neumaier": lambda: LeeMyklandEngine("rows", K=30, sum_kind="neumaier"),
        "LM clock window": lambda: LeeMyklandEngine("clock", window=100.0),
        "LM decay": lambda: LeeMyklandEngine("decay", lam=0.99),
    }
    for name, make in makers.items():
        whole = make()
        outs_whole = []
        for i in range(n):
            outs_whole.append(whole.step(float(clocks[i]), float(r[i]), 1.0))
        a = make()
        outs = []
        for i in range(n):
            if i in (n // 3, 2 * n // 3):  # a chunk edge: save, load, go on
                a = pickle.loads(pickle.dumps(copy.deepcopy(a)))
            outs.append(a.step(float(clocks[i]), float(r[i]), 1.0))
        same = all(
            all((x[k] == y[k]) or (x[k] != x[k] and y[k] != y[k]) for k in x)
            for x, y in zip(outs, outs_whole, strict=True)
        )
        resume.append({"form": name, "rows": n, "bitwise_after_save_and_load": bool(same)})
    res["resume"] = resume
    res["summary"] = {"seconds": time.time() - t0}
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"part3 done in {time.time() - t0:.1f} s -> {out}")
    print_part3(res)


def print_part3(res: dict) -> None:
    for k in (
        "state_measured",
        "state_minimal",
        "onepass",
        "vs_oracles",
        "decay",
        "windows",
        "windows_vs_batch",
        "zero_weight",
        "lm_rows",
        "window_sum_outlier",
        "lm_other",
        "bounded",
        "resume",
    ):
        print(f"\n## {k}")
        for row in res[k]:
            print(json.dumps(row, default=str))


# --------------------------------------------------------------------------
# Part 4: the Hawkes intensity (hfvol) against tick, the online form, the caps
# --------------------------------------------------------------------------
#
# Univariate Hawkes process, exponential kernel (Hawkes 1971; Ozaki 1979):
#   lambda(t) = mu + alpha sum_{t_j < t} exp(-beta (t - t_j))   (the left limit; strictly earlier)
#   log L = sum_i log lambda(t_i) - mu T - (alpha/beta) sum_i (1 - exp(-beta (T - t_i)))
# hfvol: HawkesParams(mu, alpha, beta), alpha the jump in intensity per event.
# tick:  phi(t) = a beta exp(-beta t), a the adjacency (= the branching ratio alpha/beta).
# The map: alpha = a beta.  tick's `score` for gofit="likelihood" is (log L + T)/n.


class HawkesEngine:
    """The intensity before each event as a row machine: three numbers of state.

    E = sum over events strictly before the last time of exp(-beta (last - t_j)) w_j, m = the
    weight of the events at the last time (they excite later times only), last = that time.
    An event of weight 0 advances the clock and excites nothing.
    """

    def __init__(self, mu: float, alpha: float, beta: float):
        self.mu, self.alpha, self.beta = mu, alpha, beta
        self.E = 0.0
        self.m = 0.0
        self.last: float | None = None

    def step(self, t: float, w: float = 1.0) -> float:
        """lambda(t-) from the events before t, then the event takes its place."""
        if self.last is not None and t < self.last:
            raise ValueError("the clock went backwards")
        if self.last is not None and t > self.last:
            self.E = math.exp(-self.beta * (t - self.last)) * (self.E + self.m)
            self.m = 0.0
        self.last = t
        lam = self.mu + self.alpha * self.E
        if w > 0.0 and math.isfinite(w):
            self.m += w
        return lam

    def state_bytes(self) -> int:
        return 3 * 8  # E, m, last (the parameters are the model's, not its state)


def hawkes_def_float(ev: np.ndarray, p, t: np.ndarray) -> np.ndarray:
    """lambda(t-) by the definition in float64, one sum per time (strictly earlier events)."""
    out = np.empty(len(t))
    for i, ti in enumerate(t):
        k = np.searchsorted(ev, ti, side="left")  # events strictly before ti
        out[i] = p[0] + p[1] * math.fsum(np.exp(-p[2] * (ti - ev[:k])))
    return out


def hawkes_def_exact(ev: np.ndarray, p, ti: float):
    import mpmath as mp

    mp.mp.dps = 40
    k = int(np.searchsorted(ev, ti, side="left"))
    s = mp.fsum(mp.exp(-mp.mpf(p[2]) * (mp.mpf(float(ti)) - mp.mpf(float(e)))) for e in ev[:k])
    return mp.mpf(p[0]) + mp.mpf(p[1]) * s


def hawkes_ll_exact(ev: np.ndarray, T: float, p):
    """The log-likelihood in 40 digits (strictly earlier events excite)."""
    import mpmath as mp

    mp.mp.dps = 40
    mu, al, be = (mp.mpf(x) for x in p)
    tot = mp.mpf(0)
    R = mp.mpf(0)
    prev = None
    m = 0
    for t in ev:
        t = mp.mpf(float(t))
        if prev is None:
            R = mp.mpf(0)
        elif t == prev:
            pass
        else:
            R = mp.exp(-be * (t - prev)) * (R + m)
            m = 0
        m += 1
        prev = t
        tot += mp.log(mu + al * R)
    integral = mu * T + (al / be) * mp.fsum(
        1 - mp.exp(-be * (mp.mpf(T) - mp.mpf(float(t)))) for t in ev
    )
    return tot - integral


def part4(force: bool) -> None:
    import mpmath as mp

    out = CACHE / "part4.json"
    if out.exists() and not force:
        print(f"{out} exists; not recomputing (use --force)")
        return
    import hfvol
    from hfvol import HawkesParams
    from scipy.optimize import minimize, minimize_scalar
    from tick.hawkes import HawkesExpKern, HawkesKernelExp, SimuHawkes

    t0 = time.time()
    res: dict = {
        "versions": {
            "hfvol": md.version("hfvol"),
            "tick": md.version("tick"),
            "numpydoc": md.version("numpydoc"),
        }
    }
    T = 2000.0
    mu, alpha, beta = 0.5, 0.6, 1.0
    ev1 = hfvol.hawkes_simulate(HawkesParams(mu, alpha, beta), T, rng=1)
    sim = SimuHawkes(
        kernels=[[HawkesKernelExp(intensity=alpha / beta, decay=beta)]],
        baseline=[mu],
        end_time=T,
        seed=2,
        verbose=False,
    )
    sim.track_intensity(0.05)
    sim.simulate()
    ev2 = np.asarray(sim.timestamps[0])
    tr_i, tr_t = np.asarray(sim.tracked_intensity[0]), np.asarray(sim.intensity_tracked_times)
    ev3 = np.round(ev1, 1)  # binned: tied time stamps
    res["streams"] = {
        "hfvol_simulated": len(ev1),
        "tick_simulated": len(ev2),
        "binned_ties": len(ev3),
        "tied_events": int(len(ev3) - len(np.unique(ev3))),
    }

    # ---- the intensity: hfvol and tick against the definition, at tick's own times
    rng = np.random.default_rng(4)

    def tick_fit(ev, decays, gofit="likelihood"):
        m = HawkesExpKern(
            decays=decays,
            gofit=gofit,
            solver="bfgs" if gofit == "likelihood" else "agd",
            penalty="l2",
            C=1e12,
            max_iter=20000,
            tol=1e-14,
        )
        m.fit([ev])
        return m

    inten = []
    for name, ev in (("hfvol_simulated", ev1), ("tick_simulated", ev2), ("binned_ties", ev3)):
        m = tick_fit(ev, beta)
        mu_t, a_t = float(m.baseline[0]), float(m.adjacency[0, 0])
        p = (mu_t, a_t * beta, beta)
        x, times = m.estimated_intensity([ev], 0.05, end_time=float(ev[-1]))
        x, times = np.asarray(x[0]), np.asarray(times)
        h_left = hfvol.hawkes_intensity_path(ev, HawkesParams(*p), times, include_current=False)
        h_right = hfvol.hawkes_intensity_path(ev, HawkesParams(*p), times, include_current=True)
        smp = np.sort(rng.choice(len(times), 60, replace=False))
        ex = [hawkes_def_exact(ev, p, times[i]) for i in smp]
        d = {
            "stream": name,
            "fitted_params_tick": [mu_t, a_t],
            "points": len(times),
            "tick_vs_hfvol_left_max_abs": float(np.max(np.abs(x - h_left))),
            "tick_vs_hfvol_left_max_rel": float(np.max(np.abs(x - h_left) / h_left)),
            "tick_vs_hfvol_right_max_abs": float(np.max(np.abs(x - h_right))),
            "points_where_right_differs": int(np.sum(np.abs(x - h_right) > 1e-9)),
            "tick_ulps_vs_exact_max": max(
                _ulp_gap(float(x[i]), e) for i, e in zip(smp, ex, strict=True)
            ),
            "hfvol_left_ulps_vs_exact_max": max(
                _ulp_gap(float(h_left[i]), e) for i, e in zip(smp, ex, strict=True)
            ),
        }
        # at event times only (the predictable intensity before each event)
        ie = np.flatnonzero(np.isin(times, ev))
        d["event_times_in_tick_grid"] = int(len(ie))
        d["tick_vs_hfvol_at_events_max_rel"] = (
            float(np.max(np.abs(x[ie] - h_left[ie]) / h_left[ie])) if len(ie) else None
        )
        inten.append(d)
    res["intensity"] = inten

    # tick's tracked intensity from its simulation, against the definition at the true parameters
    h_true = hfvol.hawkes_intensity_path(
        ev2, HawkesParams(mu, alpha, beta), tr_t, include_current=True
    )
    h_true_l = hfvol.hawkes_intensity_path(
        ev2, HawkesParams(mu, alpha, beta), tr_t, include_current=False
    )
    res["tick_simulation_tracked"] = {
        "points": len(tr_t),
        "max_abs_vs_hfvol_right": float(np.max(np.abs(tr_i - h_true))),
        "max_abs_vs_hfvol_left": float(np.max(np.abs(tr_i - h_true_l))),
        "max_rel_vs_hfvol_right": float(np.max(np.abs(tr_i - h_true) / h_true)),
    }

    # ---- the log-likelihood: hfvol, tick's score and the definition, at several parameters
    ll = []
    for name, ev in (("hfvol_simulated", ev1), ("binned_ties", ev3)):
        Tend = float(ev[-1])
        for pr in (
            (0.5, 0.6, 1.0),
            (0.4, 0.3, 1.0),
            (0.8, 0.0, 1.0),
            (0.45, 0.5, 0.8),
            (0.6, 1.2, 2.0),
        ):
            m = HawkesExpKern(decays=pr[2], gofit="likelihood")
            s = m.score(
                [ev], Tend, baseline=np.array([pr[0]]), adjacency=np.array([[pr[1] / pr[2]]])
            )
            n = len(ev)
            mapped = n * s - Tend  # tick's score is (log L + T)/n
            h = hfvol.hawkes_log_likelihood(ev, Tend, HawkesParams(*pr))
            e = hawkes_ll_exact(ev, Tend, pr)
            ll.append(
                {
                    "stream": name,
                    "params": pr,
                    "n": n,
                    "hfvol": h,
                    "tick_mapped": mapped,
                    "exact": float(e),
                    "hfvol_vs_exact_abs": float(abs(mp.mpf(h) - e)),
                    "tick_vs_exact_abs": float(abs(mp.mpf(mapped) - e)),
                    "tick_vs_hfvol_abs": abs(mapped - h),
                    "scale": abs(h),
                }
            )
    res["loglik"] = ll

    # ---- the fit: tick's likelihood and least-squares fits, hfvol's likelihood at a fixed beta
    fits = []
    for name, ev in (("hfvol_simulated", ev1), ("tick_simulated", ev2)):
        Tend = float(ev[-1])
        m = tick_fit(ev, beta)
        ml = tick_fit(ev, beta, "least-squares")

        def f(x, ev=ev, Tend=Tend):
            return -hfvol.hawkes_log_likelihood(ev, Tend, HawkesParams(x[0], x[1], beta))

        r = minimize(
            f,
            [0.4, 0.4],
            method="Nelder-Mead",
            options={"xatol": 1e-13, "fatol": 1e-13, "maxiter": 4000},
        )

        # profile over beta with tick: the likelihood at tick's fitted parameters, mapped
        def prof(b, ev=ev, Tend=Tend):
            mm = tick_fit(ev, float(b))
            sc = mm.score([ev], Tend, baseline=mm.baseline, adjacency=mm.adjacency)
            return -(len(ev) * sc - Tend)

        rb = minimize_scalar(prof, bounds=(0.3, 3.0), method="bounded", options={"xatol": 1e-9})
        mb = tick_fit(ev, float(rb.x))
        hp, hll = hfvol.fit_hawkes_mle(ev, Tend, n_restarts=10, rng=0)
        fits.append(
            {
                "stream": name,
                "n": len(ev),
                "tick_likelihood_fit": [float(m.baseline[0]), float(m.adjacency[0, 0] * beta)],
                "hfvol_likelihood_max_fixed_beta": [float(r.x[0]), float(r.x[1])],
                "gap_mu": abs(float(m.baseline[0]) - float(r.x[0])),
                "gap_alpha": abs(float(m.adjacency[0, 0] * beta) - float(r.x[1])),
                "tick_least_squares_fit": [float(ml.baseline[0]), float(ml.adjacency[0, 0] * beta)],
                "tick_free_beta": {
                    "beta": float(rb.x),
                    "mu": float(mb.baseline[0]),
                    "alpha": float(mb.adjacency[0, 0] * rb.x),
                    "loglik": float(-rb.fun),
                },
                "hfvol_free_beta": {"beta": hp.beta, "mu": hp.mu, "alpha": hp.alpha, "loglik": hll},
                "true": [mu, alpha, beta],
            }
        )
    res["fits"] = fits
    print(f"hawkes oracles {time.time() - t0:.1f} s")

    # ---- the online form
    eng = HawkesEngine(mu, alpha, beta)
    lam_on = np.array([eng.step(float(t), 1.0) for t in ev1])
    lam_def = hawkes_def_float(ev1, (mu, alpha, beta), ev1)
    lam_hf = hfvol.hawkes_intensity_path(
        ev1, HawkesParams(mu, alpha, beta), ev1, include_current=False
    )
    smp = np.sort(rng.choice(len(ev1), 60, replace=False))
    ex = [hawkes_def_exact(ev1, (mu, alpha, beta), ev1[i]) for i in smp]
    # ties
    eng3 = HawkesEngine(mu, alpha, beta)
    lam3 = np.array([eng3.step(float(t), 1.0) for t in ev3])
    lam3_hf = hfvol.hawkes_intensity_path(
        ev3, HawkesParams(mu, alpha, beta), ev3, include_current=False
    )
    # zero-weight events (weight 0, garbage time stamps between the real ones, same clock)
    zr = np.random.default_rng(6)
    extra = np.sort(zr.uniform(0, T, 800))
    allt = np.concatenate([ev1, extra])
    ww = np.concatenate([np.ones(len(ev1)), np.zeros(len(extra))])
    o = np.argsort(allt, kind="stable")
    engz = HawkesEngine(mu, alpha, beta)
    lz = np.array([engz.step(float(allt[i]), float(ww[i])) for i in o])
    real = ww[o] > 0
    # the intensity at each real event, with the zero-weight events present, against without
    zero_gap_ulps = float(np.max(np.abs(lz[real] - lam_on) / np.spacing(lam_on)))
    res["online"] = {
        "events": len(ev1),
        "state_bytes": eng.state_bytes(),
        "vs_definition_float_max_ulps": float(
            np.max(np.abs(lam_on - lam_def) / np.spacing(lam_def))
        ),
        "vs_hfvol_left_max_ulps": float(np.max(np.abs(lam_on - lam_hf) / np.spacing(lam_hf))),
        "bitwise_equal_to_hfvol_count": int(np.sum(lam_on == lam_hf)),
        "vs_exact_max_ulps_60_events": max(
            _ulp_gap(float(lam_on[i]), e) for i, e in zip(smp, ex, strict=True)
        ),
        "ties_vs_hfvol_max_ulps": float(np.max(np.abs(lam3 - lam3_hf) / np.spacing(lam3_hf))),
        "ties_events": int(len(ev3) - len(np.unique(ev3))),
        "zero_weight_events_inserted": len(extra),
        "zero_weight_vs_without_max_ulps": zero_gap_ulps,
        "zero_weight_state_finite": bool(all(math.isfinite(x) for x in (engz.E, engz.m))),
    }

    # ---- per-group cap: the rows each form can hold
    cap = 256 * 1024
    warn = 128 * 1024

    def rows(fixed, per, limit):
        return int((limit - fixed) // per)

    res["caps"] = {
        "cap_bytes": cap,
        "warn_bytes": warn,
        "forms": [
            {
                "form": "BNS engine, snapshot ring",
                "fixed": 89 + 56,
                "per_row": 56,
                "rows_at_warn": rows(89 + 56, 56, warn),
                "rows_at_cap": rows(89 + 56, 56, cap),
            },
            {
                "form": "BNS engine, term ring",
                "fixed": 137,
                "per_row": 48,
                "rows_at_warn": rows(137, 48, warn),
                "rows_at_cap": rows(137, 48, cap),
            },
            {
                "form": "BV alone, term ring (minimal)",
                "fixed": 32,
                "per_row": 16,
                "rows_at_warn": rows(32, 16, warn),
                "rows_at_cap": rows(32, 16, cap),
            },
            {
                "form": "BNS ratio (TQ), term ring (minimal)",
                "fixed": 64,
                "per_row": 40,
                "rows_at_warn": rows(64, 40, warn),
                "rows_at_cap": rows(64, 40, cap),
            },
            {
                "form": "Lee-Mykland K rows, compensated",
                "fixed": 48,
                "per_row": 8,
                "rows_at_warn": rows(48, 8, warn),
                "rows_at_cap": rows(48, 8, cap),
            },
            {
                "form": "Lee-Mykland clock window, term ring",
                "fixed": 80,
                "per_row": 24,
                "rows_at_warn": rows(80, 24, warn),
                "rows_at_cap": rows(80, 24, cap),
            },
            {
                "form": "Hawkes, exponential kernel",
                "fixed": 24,
                "per_row": 0,
                "rows_at_warn": None,
                "rows_at_cap": None,
            },
        ],
    }
    res["summary"] = {"seconds": time.time() - t0}
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"part4 done in {time.time() - t0:.1f} s -> {out}")
    for k, v in res.items():
        print(f"\n## {k}")
        if isinstance(v, list):
            for row in v:
                print(json.dumps(row, default=str))
        else:
            print(json.dumps(v, default=str, indent=1))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=["part1", "part2", "part3", "part4"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)
    {"part1": part1, "part2": part2, "part3": part3, "part4": part4}[a.part](a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
