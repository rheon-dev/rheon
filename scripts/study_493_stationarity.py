"""Study 493, request A18: the covariance stationarity of ADCC's ``Q`` process
(Cappiello, Engle and Sheppard 2006), derived and then settled by simulation
near the boundary. The record is ``docs/records/STUDY-493-488.md``, section 9.

    Q_{t+1} = C + a z_t z_t' + g n_t n_t' + b Q_t,   C = (1 - a - b) Qbar - g N
    R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2),  z_t = R_t^(1/2) eps_t,  n_t = min(z_t, 0)

``eps_t`` iid spherical with unit covariance: standard normal, or Student-t(5)
scaled to unit variance. ``Qbar`` has unit diagonal and off-diagonal 0.5;
``N`` is part 3 of study 493's calibrated ``N0`` (the process's own ``E[n n']``
at the truth ``(0.03, 0.93, 0.03)``; ``.cache/study_493_488/adcc_n0.json``),
held fixed ("N treated as known").

Subcommands (each caches its result as JSON under ``.cache/study_493_488/``,
so a rerun resumes or only prints):

    .venv-493/bin/python scripts/study_493_stationarity.py part1
        the derivation's numbers: delta at each (k, noise), the normal orthant
        moment E[n_i n_j] in closed form against scipy's quadrature, the
        fourth moment kappa from scipy.stats, the exact stationary mean and
        variance of a diagonal entry of Q, and the boundary grid that parts 2
        and 3 simulate (a, b, g, a + b + delta g, the intercept's eigenvalues)

    .venv-493/bin/python scripts/study_493_stationarity.py part2
        2 series, normal and t(5), the 18 grid points, 8 seeds, 100,000 rows,
        Qbar and N fixed ("fixed") and as running means ("running"):
        stationarity_k2.jsonl; then the fixed variant at four outside points
        for 1,000,000 rows: stationarity_long_k2.jsonl (a 100,000-row batch of
        144 paths takes about 19 s, a 1,000,000-row batch of 32 about 80 s;
        run it under nohup)

Environment: the venv ``.venv-493`` (numpy, scipy), excluded through
``.git/info/exclude``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

OUT = os.path.join(".cache", "study_493_488")
N0_PATH = os.path.join(OUT, "adcc_n0.json")
RHO_BAR = 0.5
NU = 5.0
TRUTH = (0.03, 0.93, 0.03)
NOISES = ("normal", "t5")
# offsets of a + b + delta g from 1: three inside, three outside
OFFSETS = (-0.01, -0.003, -0.001, 0.001, 0.003, 0.01)


def qbar_of(k: int) -> np.ndarray:
    q = np.full((k, k), RHO_BAR)
    np.fill_diagonal(q, 1.0)
    return q


def load_json(path: str):  # noqa: ANN201
    with open(path) as fh:
        return json.load(fh)


def n0_of(k: int, noise: str) -> np.ndarray:
    store = load_json(N0_PATH)
    key = f"k{k}_nu{'inf' if noise == 'normal' else int(NU)}"
    return np.array(store[key]["n0"])


def delta_of(qbar: np.ndarray, nmat: np.ndarray) -> float:
    """The largest eigenvalue of ``Qbar^(-1/2) N Qbar^(-1/2)`` (by the
    Cholesky factor: ``L^-1 N L^-T`` is similar to it)."""
    li = np.linalg.inv(np.linalg.cholesky(qbar))
    return float(np.linalg.eigvalsh(li @ nmat @ li.T)[-1])


def kappa(noise: str) -> float:
    """``E eps_1^4`` from scipy.stats (unit variance): 3 for the normal; for
    the t, ``(excess + 3)`` of the t(nu) law (scaling cancels in the ratio)."""
    from scipy import stats

    if noise == "normal":
        return float(stats.norm.stats(moments="k")) + 3.0
    return float(stats.t(NU).stats(moments="k")) + 3.0


def orthant_closed(rho: float) -> float:
    """``E[X Y 1(X < 0, Y < 0)]`` for a standard bivariate normal with
    correlation ``rho``: ``(sqrt(1 - rho^2) + rho (pi - arccos rho)) / (2 pi)``."""
    return (math.sqrt(1.0 - rho * rho) + rho * (math.pi - math.acos(rho))) / (2.0 * math.pi)


def orthant_scipy(rho: float) -> float:
    """The same by scipy's quadrature of the density (the oracle)."""
    from scipy import integrate, stats

    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    val, _ = integrate.dblquad(
        lambda y, x: x * y * mvn.pdf([x, y]), -12.0, 0.0, -12.0, 0.0, epsabs=1e-13, epsrel=1e-12
    )
    return float(val)


def diag_moments(a: float, b: float, g: float, kap: float, c_ii: float) -> dict:
    """A diagonal entry is the scalar AR(1) ``q' = c_ii + u + b q`` with iid
    input ``u = z^2 (a + g 1(z < 0))``, ``z`` of the law of ``eps_1``:
    ``E u = a + g/2``, ``Var u = kappa (a^2 + a g + g^2/2) - (a + g/2)^2``."""
    eu = a + g / 2.0
    vu = kap * (a * a + a * g + g * g / 2.0) - eu * eu
    return {"mean": (c_ii + eu) / (1.0 - b), "var": vu / (1.0 - b * b), "var_u": vu}


def grid(k: int, noise: str) -> list[dict]:
    """The boundary points: ``s = a + b + delta g - 1`` in OFFSETS along three
    directions. 'g': a, b at the truth, g moves (a + b < 1 throughout, only
    delta g crosses). 'a': b, g at the truth, a moves. 'b': a slow process,
    b = 0.98, a = 0.005, g moves."""
    qbar, nmat = qbar_of(k), n0_of(k, noise)
    d = delta_of(qbar, nmat)
    kap = kappa(noise)
    rows = []
    for direction in ("g", "a", "b"):
        for s in OFFSETS:
            if direction == "g":
                a, b = TRUTH[0], TRUTH[1]
                g = (1.0 + s - a - b) / d
            elif direction == "a":
                b, g = TRUTH[1], TRUTH[2]
                a = 1.0 + s - b - d * g
            else:
                a, b = 0.005, 0.98
                g = (1.0 + s - a - b) / d
            c = (1.0 - a - b) * qbar - g * nmat
            ev = np.linalg.eigvalsh(c)
            # the diagonal entry with the smallest C_ii (N's diagonal is not
            # exactly 1/2: N0 is a sample mean), so the smallest stationary mean
            dm = diag_moments(a, b, g, kap, float(np.min(np.diag(c))))
            rows.append(
                {
                    "k": k,
                    "noise": noise,
                    "direction": direction,
                    "offset": s,
                    "a": a,
                    "b": b,
                    "g": g,
                    "sum": a + b + d * g,
                    "delta": d,
                    "c_eig_min": float(ev[0]),
                    "c_eig_max": float(ev[-1]),
                    "c_diag_min": float(np.min(np.diag(c))),
                    "diag_mean": dm["mean"],
                    "diag_var": dm["var"],
                }
            )
    return rows


def part1(force: bool) -> dict:
    path = os.path.join(OUT, "stationarity_part1.json")
    if os.path.exists(path) and not force:
        return load_json(path)
    res: dict = {"delta": {}, "kappa": {}, "orthant": [], "grid": {}, "truth_diag": {}}
    for k in (2, 3):
        for noise in NOISES:
            qbar, nmat = qbar_of(k), n0_of(k, noise)
            res["delta"][f"k{k}_{noise}"] = delta_of(qbar, nmat)
            res["grid"][f"k{k}_{noise}"] = grid(k, noise)
    for noise in NOISES:
        res["kappa"][noise] = kappa(noise)
        a, b, g = TRUTH
        c_ii = (1.0 - a - b) - g * 0.5
        res["truth_diag"][noise] = diag_moments(a, b, g, res["kappa"][noise], c_ii)
    for rho in (-0.5, 0.0, 0.3, 0.5, 0.9):
        res["orthant"].append(
            {
                "rho": rho,
                "closed": orthant_closed(rho),
                "scipy": orthant_scipy(rho),
                "record_start_formula": rho / 4.0 + math.sqrt(1.0 - rho * rho) / (2.0 * math.pi),
            }
        )
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def print_part1(res: dict) -> None:
    print("delta:", {k: round(v, 4) for k, v in res["delta"].items()})
    print("kappa:", res["kappa"])
    print("diagonal moments at the truth:", res["truth_diag"])
    for o in res["orthant"]:
        print(
            f"orthant rho={o['rho']:+.1f}: closed {o['closed']:.12f} "
            f"scipy {o['scipy']:.12f} diff {abs(o['closed'] - o['scipy']):.1e} "
            f"record start {o['record_start_formula']:.4f}"
        )
    for key, rows in res["grid"].items():
        print(key)
        for r in rows:
            print(
                f"  {r['direction']} {r['offset']:+.3f}: a {r['a']:.4f} b {r['b']:.4f} "
                f"g {r['g']:.4f} sum {r['sum']:.4f} "
                f"C eig [{r['c_eig_min']:+.5f}, {r['c_eig_max']:.5f}] "
                f"C_ii min {r['c_diag_min']:+.5f} diag mean {r['diag_mean']:.4f} "
                f"var {r['diag_var']:.5f}"
            )


# --------------------------------------------------------------------------
# parts 2 and 3: the simulation near the boundary

SIM_SEEDS = tuple(range(493100, 493108))  # 8 seeds; each seed's noise is shared by every grid point
SIM_ROWS = 100_000
CHECKPOINTS = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
LONG_ROWS = 1_000_000
# the outside points that did not leave the cone in 100,000 rows of the fixed variant
LONG_POINTS = (("g", 0.003), ("g", 0.01), ("a", 0.003), ("b", 0.003))
N_BLOCKS = 10
RUN_PRIOR = 500  # the running variant's Qbar and N start as 500 pseudo-rows at Qbar, N0


def simulate_batch(
    k: int, noise: str, variant: str, rows: int = SIM_ROWS, select: tuple | None = None
) -> dict:
    """Every grid point of ``part1`` at ``k`` and ``noise``, every seed, one
    path each, ``rows`` rows, vectorized over the paths (G points x S seeds).

    ``variant="fixed"``: ``Qbar``, ``N`` fixed (the derivation's case).
    ``variant="running"``: ``Qbar_t``, ``N_t`` the running means of ``z z'``
    and ``n n'`` (from 500 pseudo-rows at ``Qbar``, ``N0``), the engine's form.

    The state follows the recursion as written whatever happens. When ``Q_t``
    is not PSD (the model is undefined there), ``z_t`` is drawn from the
    eigenvalue-clipped ``Q_t`` normalized to unit diagonal (§9.5's ``Q~``):
    the path is marked as having left the cone at its first such row, and
    continues so that its moments can still be watched."""
    g_rows = grid(k, noise)
    if select is not None:
        g_rows = [r for r in g_rows if (r["direction"], r["offset"]) in select]
    qbar, nmat = qbar_of(k), n0_of(k, noise)
    G, S = len(g_rows), len(SIM_SEEDS)
    a = np.array([r["a"] for r in g_rows])[:, None, None, None]
    b = np.array([r["b"] for r in g_rows])[:, None, None, None]
    g = np.array([r["g"] for r in g_rows])[:, None, None, None]
    iu = np.triu_indices(k)
    ne = len(iu[0])
    rngs = [np.random.default_rng(s) for s in SIM_SEEDS]
    q = np.broadcast_to(qbar, (G, S, k, k)).copy()
    qb_run = np.broadcast_to(qbar, (G, S, k, k)).copy()
    n_run = np.broadcast_to(nmat, (G, S, k, k)).copy()
    c_fixed = (1.0 - a - b) * qbar - g * nmat
    s1 = np.zeros((G, S, ne))
    s2 = np.zeros((G, S, ne))
    blk1 = np.zeros((N_BLOCKS, G, S, ne))
    blk2 = np.zeros((N_BLOCKS, G, S, ne))
    ck = {}
    first_out = np.full((G, S), -1, dtype=np.int64)
    n_out = np.zeros((G, S), dtype=np.int64)
    lam_min = np.full((G, S), np.inf)
    diag_min = np.full((G, S), np.inf)
    blk_len = rows // N_BLOCKS
    chunk = 10_000
    for t0 in range(0, rows, chunk):
        m = min(chunk, rows - t0)
        e = np.stack([r.standard_normal((m, k)) for r in rngs], axis=1)  # (m, S, k)
        if noise == "t5":
            w = np.stack([np.sqrt((NU - 2.0) / r.chisquare(NU, m)) for r in rngs], axis=1)
            e = e * w[:, :, None]
        for j in range(m):
            t = t0 + j
            ent = q[..., iu[0], iu[1]]
            s1 += ent
            s2 += ent * ent
            bi = t // blk_len
            blk1[bi] += ent
            blk2[bi] += ent * ent
            lam, vec = np.linalg.eigh(q)
            lmin = lam[..., 0]
            dmin = np.min(np.diagonal(q, axis1=-2, axis2=-1), axis=-1)
            out = (lmin < 0.0) | (dmin <= 0.0)
            n_out += out
            first_out[(first_out < 0) & out] = t
            np.minimum(lam_min, lmin, out=lam_min)
            np.minimum(diag_min, dmin, out=diag_min)
            # z = D^(-1/2) Q+^(1/2) eps, Q+ the clipped Q, D its diagonal: cov z = R
            lamc = np.maximum(lam, 1e-12 * np.maximum(lam[..., -1:], 1e-300))
            root = (vec * np.sqrt(lamc)[..., None, :]) @ np.swapaxes(vec, -1, -2)
            dq = np.einsum("...ij,...ij->...i", vec * lamc[..., None, :], vec)
            z = np.einsum("gsij,sj->gsi", root, e[j]) / np.sqrt(dq)
            nv = np.minimum(z, 0.0)
            zz = z[..., :, None] * z[..., None, :]
            nn = nv[..., :, None] * nv[..., None, :]
            if variant == "fixed":
                c = c_fixed
            else:
                c = (1.0 - a - b) * qb_run - g * n_run
                cnt = RUN_PRIOR + t + 1
                qb_run += (zz - qb_run) / cnt
                n_run += (nn - n_run) / cnt
            q = c + a * zz + g * nn + b * q
            if t + 1 in CHECKPOINTS:
                mu = s1 / (t + 1)
                ck[t + 1] = {"mean": mu, "var": s2 / (t + 1) - mu * mu}
    bmu = blk1 / blk_len
    bvar = blk2 / blk_len - bmu * bmu
    res = {"k": k, "noise": noise, "variant": variant, "rows": rows, "seeds": list(SIM_SEEDS)}
    res["entries"] = [f"q{i + 1}{j + 1}" for i, j in zip(*iu, strict=True)]
    res["points"] = []
    for gi, r in enumerate(g_rows):
        p = dict(r)
        p["first_out"] = first_out[gi].tolist()
        p["frac_rows_out"] = (n_out[gi] / rows).tolist()
        p["lam_min"] = lam_min[gi].tolist()
        p["diag_min"] = diag_min[gi].tolist()
        p["checkpoints"] = {
            str(t): {"mean": v["mean"][gi].tolist(), "var": v["var"][gi].tolist()}
            for t, v in ck.items()
        }
        p["block_mean"] = bmu[:, gi].tolist()  # (blocks, seeds, entries)
        p["block_var"] = bvar[:, gi].tolist()
        # the exact diagonal moments, entry by entry (N0's diagonal is not exactly 1/2)
        kap = kappa(noise)
        ex = []
        for i in range(k):
            cii = (1.0 - r["a"] - r["b"]) * qbar[i, i] - r["g"] * nmat[i, i]
            ex.append(diag_moments(r["a"], r["b"], r["g"], kap, cii))
        p["diag_exact"] = ex
        res["points"].append(p)
    return res


def sim_part(k: int, force: bool) -> list[dict]:
    """Both noises and both variants at ``k``; each batch appended to
    ``stationarity_k<k>.jsonl`` as it finishes (a rerun skips the done ones)."""
    path = os.path.join(OUT, f"stationarity_k{k}.jsonl")
    done: dict = {}
    if os.path.exists(path) and not force:
        with open(path) as fh:
            for line in fh:
                d = json.loads(line)
                done[(d["noise"], d["variant"])] = d
    for noise in NOISES:
        for variant in ("fixed", "running"):
            if (noise, variant) in done:
                continue
            t0 = time.time()
            d = simulate_batch(k, noise, variant)
            d["seconds"] = time.time() - t0
            print(f"k={k} {noise} {variant}: {d['seconds']:.1f} s", flush=True)
            with open(path, "a") as fh:
                fh.write(json.dumps(d) + "\n")
            done[(noise, variant)] = d
    return [done[(n, v)] for n in NOISES for v in ("fixed", "running")]


def long_part(k: int, force: bool) -> list[dict]:
    """The fixed variant at LONG_POINTS for 1,000,000 rows, both noises;
    appended to ``stationarity_long_k<k>.jsonl``."""
    path = os.path.join(OUT, f"stationarity_long_k{k}.jsonl")
    done: dict = {}
    if os.path.exists(path) and not force:
        with open(path) as fh:
            for line in fh:
                d = json.loads(line)
                done[d["noise"]] = d
    for noise in NOISES:
        if noise in done:
            continue
        t0 = time.time()
        d = simulate_batch(k, noise, "fixed", rows=LONG_ROWS, select=LONG_POINTS)
        d["seconds"] = time.time() - t0
        print(f"long k={k} {noise}: {d['seconds']:.1f} s", flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(d) + "\n")
        done[noise] = d
    return [done[n] for n in NOISES]


def print_long(batches: list[dict]) -> None:
    for d in batches:
        print(
            f"\n== long k={d['k']} {d['noise']} fixed "
            f"({d['rows']} rows, {d.get('seconds', 0):.0f} s)"
        )
        for p in d["points"]:
            fo = np.array(p["first_out"])
            lam = np.array(p["lam_min"])
            print(
                f" {p['direction']} {p['offset']:+.3f} lamC {p['c_eig_min']:+.4f}: "
                f"left {int((fo >= 0).sum())}/{len(fo)} first out {sorted(fo[fo >= 0].tolist())} "
                f"min lam(Q) median {np.median(lam):+.4f} "
                f"range [{lam.min():+.4f}, {lam.max():+.4f}]"
            )


def summarise_point(p: dict, k: int) -> dict:
    """Per point, over seeds: paths that left the cone, first exit (median,
    range), and the settle/drift figures: for each entry, the full-path mean
    and variance (median over seeds), the change between the first and the
    second half (mean: in units of the across-seed spread of the half means;
    variance: as a ratio), and, for the diagonal, the gap to the exact mean
    in standard errors of a time average of an AR(1) with coefficient b."""
    fo = np.array(p["first_out"])
    left = fo >= 0
    bm = np.array(p["block_mean"])  # (blocks, seeds, entries)
    bv = np.array(p["block_var"])
    h = bm.shape[0] // 2
    m1, m2 = bm[:h].mean(0), bm[h:].mean(0)
    v1 = (bv[:h] + bm[:h] ** 2).mean(0) - m1**2
    v2 = (bv[h:] + bm[h:] ** 2).mean(0) - m2**2
    last = p["checkpoints"][str(SIM_ROWS)]
    mean_all = np.array(last["mean"])
    var_all = np.array(last["var"])
    out = {
        "direction": p["direction"],
        "offset": p["offset"],
        "sum": p["sum"],
        "c_eig_min": p["c_eig_min"],
        "left": int(left.sum()),
        "first_out_median": float(np.median(fo[left])) if left.any() else None,
        "first_out_min": int(fo[left].min()) if left.any() else None,
        "first_out_max": int(fo[left].max()) if left.any() else None,
        "frac_rows_out_median": float(np.median(p["frac_rows_out"])),
        "lam_min_median": float(np.median(p["lam_min"])),
        "mean_median": np.median(mean_all, 0).tolist(),
        "var_median": np.median(var_all, 0).tolist(),
        "half_mean_shift_max": float(np.max(np.abs(np.median(m2 - m1, 0)))),
        "half_var_ratio": np.median(v2 / v1, 0).tolist(),
    }
    # the diagonal against its exact moments
    ex = p["diag_exact"]
    diag_idx = [idx for idx, (i, j) in enumerate(zip(*np.triu_indices(k), strict=True)) if i == j]
    zs, vr = [], []
    for i, idx in enumerate(diag_idx):
        mu, var, bb = ex[i]["mean"], ex[i]["var"], p["b"]
        se = math.sqrt(var * (1.0 + bb) / (1.0 - bb) / SIM_ROWS)
        zs.extend(((mean_all[:, idx] - mu) / se).tolist())
        vr.extend((var_all[:, idx] / var).tolist())
    out["diag_mean_z_maxabs"] = float(np.max(np.abs(zs)))
    out["diag_var_ratio_range"] = [float(np.min(vr)), float(np.max(vr))]
    out["diag_exact_var"] = ex[0]["var"]
    # the running variance of the off-diagonal over the checkpoints (median over seeds)
    off = [idx for idx in range(mean_all.shape[1]) if idx not in diag_idx]
    out["offdiag_var_by_checkpoint"] = [
        float(np.median(np.array(p["checkpoints"][str(t)]["var"])[:, off]))
        for t in CHECKPOINTS
        if str(t) in p["checkpoints"]
    ]
    out["offdiag_mean_by_checkpoint"] = [
        float(np.median(np.array(p["checkpoints"][str(t)]["mean"])[:, off]))
        for t in CHECKPOINTS
        if str(t) in p["checkpoints"]
    ]
    return out


def print_sim(batches: list[dict]) -> None:
    for d in batches:
        k = d["k"]
        print(
            f"\n== k={k} {d['noise']} {d['variant']} ({d['rows']} rows, {len(d['seeds'])} seeds, "
            f"{d.get('seconds', 0):.0f} s)"
        )
        for p in d["points"]:
            s = summarise_point(p, k)
            fo = (
                f"first out med {s['first_out_median']:.0f} "
                f"[{s['first_out_min']}, {s['first_out_max']}]"
                if s["left"]
                else "never out"
            )
            vr = s["diag_var_ratio_range"]
            print(
                f" {s['direction']} {s['offset']:+.3f} lamC {s['c_eig_min']:+.4f}: "
                f"left {s['left']}/{len(d['seeds'])} {fo} "
                f"rows out {s['frac_rows_out_median']:.4f} | diag z {s['diag_mean_z_maxabs']:.2f} "
                f"var/exact [{vr[0]:.3f},{vr[1]:.3f}] | "
                f"off mean {s['offdiag_mean_by_checkpoint'][-1]:.4f} "
                f"var by ck {[f'{v:.4f}' for v in s['offdiag_var_by_checkpoint']]} "
                f"half shift {s['half_mean_shift_max']:.4f} var ratio "
                f"{[f'{v:.3f}' for v in s['half_var_ratio']]}"
            )


def print_lam(batches: list[dict]) -> None:
    """The smallest eigenvalue of Q over each path (median over seeds), and
    the median over seeds of the diagonal's variance against its exact value."""
    print("\nmin over the path of lambda_min(Q), median over seeds; diag var/exact median")
    for d in batches:
        cells = []
        for p in d["points"]:
            vr = []
            last = np.array(p["checkpoints"][str(d["rows"])]["var"])
            k = d["k"]
            di = [i for i, (r, c) in enumerate(zip(*np.triu_indices(k), strict=True)) if r == c]
            for i, idx in enumerate(di):
                vr.extend((last[:, idx] / p["diag_exact"][i]["var"]).tolist())
            cells.append(
                f"{p['direction']}{p['offset']:+.3f}:{np.median(p['lam_min']):+.3f}/{np.median(vr):.3f}"
            )
        print(f" {d['noise']} {d['variant']}: " + " ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("part1", "part2", "part3"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.part == "part1":
        print_part1(part1(args.force))
    elif args.part == "part2":
        batches = sim_part(2, args.force)
        print_sim(batches)
        print_lam(batches)
        print_long(long_part(2, args.force))


if __name__ == "__main__":
    sys.exit(main())
