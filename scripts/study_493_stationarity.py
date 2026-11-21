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

Environment: the venv ``.venv-493`` (numpy, scipy), excluded through
``.git/info/exclude``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("part1",))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.part == "part1":
        print_part1(part1(args.force))


if __name__ == "__main__":
    sys.exit(main())
