"""GARCH(1,1) with variance targeting: the online estimator against a batch
variance-targeting fit run by arch (docs/records/ORACLE-STUDY-485.md, request
A10 part 1).

**The batch estimator (the oracle).** Variance targeting (Engle and Mezrich
1996; Francq, Horvath and Zakoian 2011) replaces ``omega`` by the sample
variance and the persistence::

    v_T   = (1/T) sum_t r_t^2                       (zero mean: the mean of r^2)
    omega = v_T * (1 - alpha - beta)
    (alpha, beta) = argmax  sum_t l_t(omega(alpha, beta), alpha, beta)

with ``l_t`` the Gaussian log-likelihood of ``garch.py``. No arch release has
a variance-targeting process: 9.0.0 (installed) has none, and neither do
4.8.1, 4.13, 4.15, 4.17, 4.19, 5.6.0, 6.3.0, 7.2.0 or 8.0.0, whose
``volatility.py`` were searched. So the oracle is arch's own ``GARCH``
subclassed with two parameters, ``(alpha, beta)``: ``compute_variance`` sets
``omega`` as above (``v_T`` the mean of the residuals' squares it is handed,
the whole fit sample) and calls arch's compiled ``garch_recursion``. Everything
else is arch's: ``arch_model(..., mean="Zero", dist="normal", rescale=False)``
with this process, its default backcast (the 0.94-weighted mean of the first
75 ``r^2``), its variance bounds, its SLSQP fit and its ``cov_type="robust"``
(Bollerslev-Wooldridge) covariance of ``(alpha, beta)``. Those standard errors
treat ``v_T`` as known; they are not the targeted estimator's full sampling
error, which adds the variance of ``v_T`` (Francq, Horvath and Zakoian 2011).
The online and batch estimates share ``v_T`` at the last row (below), so the
gap between them carries no ``v_T`` error. The bounds are ``alpha, beta`` in
``[0, 1]`` with ``alpha + beta <= 1``; starting values are the best of arch's
own grid of ``(alpha, persistence)`` pairs by the Gaussian likelihood.

A second batch fit, by ``scipy.optimize.minimize`` (L-BFGS-B) on the same
restricted likelihood with the filter run by ``scipy.signal.lfilter``, started
at ``(0.05, 0.90)`` (not at arch's optimum), checks
that arch's SLSQP reached the maximum (``cross_*`` fields).

**The online estimator** is ``garch.OnlineGarch(target="running")``, part 3's
variant 1a, unchanged: Gauss-Newton steps on ``(alpha, beta)`` only, with
``omega = v_t * (1 - alpha - beta)`` reset after every step, ``v_t`` the
running mean of ``r^2`` over the rows read so far (the 75 buffered rows
included; before the buffer is replayed, the backcast stands in). The
full-sample ``v_T`` is not a stream estimator: it needs the last row before
the first variance is emitted. The running ``v_t`` is the streaming version.
**The difference:** at the last row ``v_t`` is ``v_T`` (the same rows; the
incremental mean differs from ``np.mean`` by rounding, ``vbar_rel_rounding``),
so ``omega_online - omega_batch = -v_T * (dalpha + dbeta)`` up to that
rounding; along the stream, the rows before ``T`` were filtered and their
steps taken with ``v_t``, not ``v_T`` (``vbar_half_rel``: ``v_{T/2} / v_T - 1``),
which is the price of being one-pass.

    python vt_garch.py OUT.jsonl            # the grid, resuming after the last finished case
    python vt_garch.py OUT.jsonl --summary  # the table (several files: their union)
    python vt_garch.py --check              # the oracle's own checks (a few seconds)

Options for the grid (request A10, part 2): ``--rows N [N ...]`` replaces the
row counts (default 2,000 and 10,000); ``--persistence P`` replaces the truth
by ``(1 - P, 0.08, P - 0.08)``: alpha stays 0.08, beta takes the rest of the
persistence and omega is set so the unconditional variance
``omega / (1 - alpha - beta)`` is 1.0, as G1's is; ``--shard I/K`` runs only
the seeds whose index in 485000..485049 is I modulo K, so K processes can share
a grid, each appending to its own file. ``--nu NU`` (part 3) draws Student-t
noise with ``NU`` degrees of freedom, scaled to unit variance, and makes arch's
``dist="t"`` fit (``nu`` estimated) the primary oracle (``compare``).

``--remedy NAME`` (request A11) runs the online estimator with one of the
early-transient remedies in ``REMEDIES`` (options of ``garch.OnlineGarch``)
in place of part 1's; the oracle is unchanged. Without it every field is as
before. ``batch500``: remedy (a), the first 500 rows fitted by the targeted
batch fit and used as the online start (``t0 = 500``; those rows emit
nothing). ``gain500``, ``gain2000``: remedy (b), the step's gain
``1/(t + T)`` at ``T`` 500 or 2,000 in place of ``1/(t + 75)``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import garch  # noqa: E402

TRUTH = (0.02, 0.08, 0.90)
SEEDS = range(485000, 485050)
SIZES = (2000, 10000)
PARAMS = ("alpha", "beta")
CROSS_START = np.array([0.05, 0.90])  # the online start, not arch's optimum
ONLINE = {"target": "running"}  # part 1's estimator (part 3's 1a)
REMEDIES: dict[str, dict[str, object]] = {
    "batch500": {**ONLINE, "batch_start": True, "warm": 500, "t0": 500.0},
    "gain500": {**ONLINE, "gain": "offset", "gain_t0": 500.0},
    "gain2000": {**ONLINE, "gain": "offset", "gain_t0": 2000.0},
}


def vt_process():  # noqa: ANN201
    """arch's GARCH(1,1) with ``omega`` targeted: two free parameters."""
    from arch.univariate import GARCH
    from arch.univariate import recursions as rec

    class VTGarch(GARCH):
        def __init__(self) -> None:
            super().__init__(p=1, o=0, q=1)
            self._num_params = 2
            self._name = "VT-GARCH"

        @staticmethod
        def target(resids: np.ndarray) -> float:
            return float(np.mean(np.asarray(resids, dtype=float) ** 2))

        def full_params(self, parameters: np.ndarray, resids: np.ndarray) -> np.ndarray:
            a, b = float(parameters[0]), float(parameters[1])
            return np.array([self.target(resids) * (1.0 - a - b), a, b])

        def compute_variance(self, parameters, resids, sigma2, backcast, var_bounds):  # noqa: ANN001, ANN201
            parameters, resids, backcast, var_bounds = self._check_compute_variance(
                parameters, resids, sigma2, backcast, var_bounds
            )
            full = self.full_params(parameters, resids)
            fresids = np.absolute(resids) ** 2.0
            sresids = np.sign(resids)
            rec.garch_recursion(
                full, fresids, sresids, sigma2, 1, 0, 1, resids.shape[0], backcast, var_bounds
            )
            return sigma2

        def bounds(self, resids):  # noqa: ANN001, ANN201
            return [(0.0, 1.0), (0.0, 1.0)]

        def constraints(self):  # noqa: ANN201
            # alpha >= 0, beta >= 0, -(alpha + beta) >= -1
            return np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]), np.array([0.0, 0.0, -1.0])

        def starting_values(self, resids):  # noqa: ANN001, ANN201
            resids = np.asarray(resids, dtype=float)
            var_bounds = self.variance_bounds(resids)
            backcast = self.backcast(resids)
            best, best_llf = None, -math.inf
            for alpha in (0.01, 0.05, 0.1, 0.2):
                for persistence in (0.5, 0.7, 0.9, 0.98):
                    sv = np.array([alpha, persistence - alpha])
                    llf = self._gaussian_loglikelihood(sv, resids, backcast, var_bounds)
                    if llf > best_llf:
                        best, best_llf = sv, llf
            return best

        def parameter_names(self) -> list[str]:
            return ["alpha[1]", "beta[1]"]

    return VTGarch()


def vt_fit(r: np.ndarray, dist: str = "normal"):  # noqa: ANN201
    """The oracle: arch's fit of the targeted process (module docstring), with
    arch's ``dist`` option: ``"normal"``, or ``"t"``, arch's ``StudentsT``,
    which estimates the degrees of freedom ``nu`` beside ``(alpha, beta)``."""
    from arch import arch_model

    am = arch_model(r, mean="Zero", vol="GARCH", dist=dist, rescale=False)
    am.volatility = vt_process()
    return am, am.fit(disp="off", cov_type="robust")


def _lfilter_path(alpha: float, beta: float, r2: np.ndarray, v: float, start: float) -> np.ndarray:
    from scipy.signal import lfilter

    omega = v * (1.0 - alpha - beta)
    x = np.empty_like(r2)
    x[0] = omega + alpha * start + beta * start
    x[1:] = omega + alpha * r2[:-1]
    return lfilter([1.0], [1.0, -beta], x)


def scipy_vt_fit(r: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, float]:
    """The cross-check: the same restricted likelihood maximized by scipy."""
    from scipy.optimize import minimize

    r2 = r * r
    v = float(np.mean(r2))
    start = garch.backcast(r2[: garch.BACKCAST_ROWS])

    # box coordinates: persistence p = alpha + beta, and alpha's share s of it
    def ab(x: np.ndarray) -> tuple[float, float]:
        return float(x[0] * x[1]), float(x[0] * (1.0 - x[1]))

    def nll(x: np.ndarray) -> float:
        a, b = ab(x)
        s2 = _lfilter_path(a, b, r2, v, start)
        return 0.5 * float(np.sum(garch.LN_2PI + np.log(s2) + r2 / s2))

    p0 = float(x0[0] + x0[1])
    res = minimize(
        nll,
        np.array([p0, float(x0[0]) / p0]),
        method="L-BFGS-B",
        bounds=[(0.0, 1.0 - 1e-9), (0.0, 1.0)],
        options={"ftol": 1e-15, "gtol": 1e-9, "maxiter": 5000},
    )
    return np.array(ab(res.x)), -float(res.fun)


def _gaps(
    prefix: str,
    res,  # noqa: ANN001
    v_T: float,
    theta_o: np.ndarray,
    path_args: tuple[np.ndarray, float, np.ndarray, slice],
) -> dict[str, float]:
    """The online parameters against one batch fit, in its robust SEs."""
    r, start, pred, tail = path_args
    ab_b = np.asarray(res.params, dtype=float)[:2]
    se_b = np.asarray(res.std_err, dtype=float)[:2]
    cov_b = np.asarray(res.param_cov, dtype=float)[:2, :2]
    omega_b = v_T * (1.0 - ab_b[0] - ab_b[1])
    # omega's SE by the delta method on arch's covariance, v_T held fixed
    grad = np.array([-v_T, -v_T])
    omega_se_b = float(math.sqrt(grad @ cov_b @ grad))
    path_b = garch.filter_variance(np.array([omega_b, *ab_b]), r, start)
    out = {
        "arch_converged": float(res.convergence_flag),
        "arch_path_vs_filter_max_rel": float(
            np.max(np.abs(path_b / res.conditional_volatility**2 - 1.0))
        ),
        "pred_median_rel_second_half": float(np.median(np.abs(pred[tail] / path_b[tail] - 1.0))),
        "omega_batch": omega_b,
        "omega_diff": float(theta_o[0] - omega_b),
        "omega_diff_from_persistence": float(
            -v_T * ((theta_o[1] - ab_b[0]) + (theta_o[2] - ab_b[1]))
        ),
        "omega_batch_se_delta": omega_se_b,
        "omega_diff_in_se": float((theta_o[0] - omega_b) / omega_se_b),
    }
    for i, name in enumerate(PARAMS):
        out[f"{name}_batch"] = float(ab_b[i])
        out[f"{name}_diff"] = float(theta_o[i + 1] - ab_b[i])
        out[f"{name}_batch_se"] = float(se_b[i])
        out[f"{name}_diff_in_se"] = float((theta_o[i + 1] - ab_b[i]) / se_b[i])
    return {prefix + k: v for k, v in out.items()}


def compare(
    n: int,
    seed: int,
    truth: tuple[float, float, float] = TRUTH,
    nu: float | None = None,
    remedy: str = "",
) -> dict[str, float]:
    """The online estimator against the batch targeting fit on one series.

    With ``nu`` the noise is Student-t with ``nu`` degrees of freedom scaled to
    unit variance, and the primary oracle is arch's fit with ``dist="t"``,
    estimating ``nu`` as arch does; the Gaussian quasi-likelihood fit, the
    online estimator's own estimand, is reported beside it (``qml_`` fields).
    Without ``nu`` the fields are part 1's, unchanged. ``remedy`` names an
    entry of ``REMEDIES``; empty, the estimator is part 1's."""
    r = garch.simulate(n, *truth, seed=seed, nu=nu)
    r2 = r * r
    am, res = vt_fit(r, "t" if nu else "normal")
    v_T = am.volatility.target(r)

    est = garch.run(r, **(REMEDIES[remedy] if remedy else ONLINE))
    theta_o = est.theta.copy()
    start = garch.backcast(r2[: garch.BACKCAST_ROWS])
    pred = np.asarray(est.pred)
    path_args = (r, start, pred, slice(n // 2, n))
    ab_o = theta_o[1:]
    llf_b = float(res.loglikelihood)
    if nu:
        nu_b = float(np.asarray(res.params)[2])
        llf_o = float(am.fix(np.array([*ab_o, nu_b])).loglikelihood)
        am_q, res_q = vt_fit(r, "normal")
    else:
        llf_o = garch.loglik(
            r, garch.filter_variance(np.array([v_T * (1 - ab_o[0] - ab_o[1]), *ab_o]), r, start)
        )
        res_q = res
    ab_s, llf_s = scipy_vt_fit(r, CROSS_START)
    ab_q = np.asarray(res_q.params, dtype=float)[:2]
    se_q = np.asarray(res_q.std_err, dtype=float)[:2]
    out = {
        "n": float(n),
        "seed": float(seed),
        "v_T": v_T,
        "vbar_rel_rounding": est.mean_r2 / v_T - 1.0,
        "vbar_half_rel": float(np.mean(r2[: n // 2])) / v_T - 1.0,
        "llf_batch": llf_b,
        "llf_online_params": llf_o,
        "llf_diff_per_row": (llf_b - llf_o) / n,
        "cross_llf_scipy_minus_arch": llf_s - float(res_q.loglikelihood),
        "omega_true": truth[0],
        "omega_online": float(theta_o[0]),
    }
    if remedy:
        out["remedy"] = remedy  # type: ignore[assignment]
    out.update(_gaps("", res, v_T, theta_o, path_args))
    if nu:
        out["nu_true"] = float(nu)
        out["nu_batch"] = nu_b
        out["nu_batch_se"] = float(np.asarray(res.std_err)[2])
        out.update(_gaps("qml_", res_q, v_T, theta_o, path_args))
    for i, name in enumerate(PARAMS):
        out[f"{name}_true"] = truth[i + 1]
        out[f"{name}_online"] = float(theta_o[i + 1])
        out[f"cross_{name}_in_se"] = float((ab_s[i] - ab_q[i]) / se_q[i])
    return out


def check() -> None:
    """The oracle's own checks: its filter is garch.py's at omega = v_T(1 - a - b),
    and its fit agrees with an independent optimizer."""
    r = garch.simulate(5000, *TRUTH, seed=11)
    vt = vt_process()
    ab = np.array([0.07, 0.91])
    start = garch.backcast(r[: garch.BACKCAST_ROWS] ** 2)
    bounds = np.tile([0.0, np.inf], (r.shape[0], 1))
    ref = vt.compute_variance(ab, r, np.empty_like(r), start, bounds)
    ours = garch.filter_variance(vt.full_params(ab, r), r, start)
    lf = _lfilter_path(ab[0], ab[1], r * r, vt.target(r), start)
    print(f"filter vs garch.py max rel {np.max(np.abs(ours / ref - 1)):.3g}")
    print(f"filter vs lfilter max rel {np.max(np.abs(lf / ref - 1)):.3g}")
    am, res = vt_fit(r)
    ab_s, llf_s = scipy_vt_fit(r, CROSS_START)
    print(res.summary().tables[1])
    print(f"scipy {ab_s}, llf diff {llf_s - res.loglikelihood:.3g}")
    print(f"arch backcast {am.volatility.backcast(r)} == garch.py {start}")


def done(out: Path) -> set[tuple[int, int]]:
    if not out.exists():
        return set()
    return {
        (int(rec["seed"]), int(rec["n"]))
        for rec in (
            json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()
        )
    }


def truth_at(persistence: float | None) -> tuple[float, float, float]:
    """G1, or alpha 0.08 at another persistence with unit unconditional variance."""
    if persistence is None:
        return TRUTH
    return (1.0 - persistence, 0.08, persistence - 0.08)


def run(
    out: Path,
    sizes: tuple[int, ...] = SIZES,
    truth: tuple[float, float, float] = TRUTH,
    shard: tuple[int, int] = (0, 1),
    nu: float | None = None,
    remedy: str = "",
) -> None:
    finished = done(out)
    seeds = [s for i, s in enumerate(SEEDS) if i % shard[1] == shard[0]]
    with out.open("a", encoding="utf-8") as f:
        for n in sizes:
            for seed in seeds:
                if (seed, n) in finished:
                    continue
                f.write(json.dumps(compare(n, seed, truth, nu, remedy)) + "\n")
                f.flush()
                print(f"vt: seed {seed} n {n} done", flush=True)


def summary(outs: list[Path]) -> None:
    recs = [
        json.loads(x)
        for out in outs
        for x in out.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    for n in sorted({int(r["n"]) for r in recs}):
        rows = [r for r in recs if int(r["n"]) == n]
        conv = sum(r["arch_converged"] == 0 for r in rows)
        truths = sorted({(r["omega_true"], r["alpha_true"], r["beta_true"]) for r in rows})
        print(f"rows {n}: {len(rows)} seeds; truth {truths}; arch converged on {conv}")
        for k in (
            "alpha_diff_in_se",
            "beta_diff_in_se",
            "omega_diff_in_se",
            "omega_diff",
            "alpha_diff",
            "beta_diff",
            "llf_diff_per_row",
            "pred_median_rel_second_half",
            "vbar_half_rel",
            "vbar_rel_rounding",
            "cross_alpha_in_se",
            "cross_beta_in_se",
            "cross_llf_scipy_minus_arch",
            "arch_path_vs_filter_max_rel",
        ):
            a = np.abs([r[k] for r in rows])
            med, p90, top = np.median(a), np.quantile(a, 0.9), np.max(a)
            print(f"  {k:30s} median {med:10.4g}  p90 {p90:10.4g}  max {top:10.4g}")
        ab = sum(all(abs(r[f"{p}_diff_in_se"]) <= 2.0 for p in PARAMS) for r in rows)
        abw = sum(all(abs(r[f"{p}_diff_in_se"]) <= 2.0 for p in (*PARAMS, "omega")) for r in rows)
        over3 = sum(any(abs(r[f"{p}_diff_in_se"]) > 3.0 for p in PARAMS) for r in rows)
        print(f"  alpha, beta within 2 SE: {ab}; with omega (delta SE) too: {abw}")
        print(f"  any of alpha, beta over 3 SE: {over3}")
        worst = sorted(rows, key=lambda r: -max(abs(r[f"{p}_diff_in_se"]) for p in PARAMS))[:3]
        for r in worst:
            print(
                f"  worst seed {int(r['seed'])}: alpha {r['alpha_diff_in_se']:+.2f} SE, "
                f"beta {r['beta_diff_in_se']:+.2f} SE, omega {r['omega_diff']:+.2e}"
            )
        verdict = "passes" if ab >= 0.9 * len(rows) else "fails"
        print(f"  criterion (90% of seeds with alpha, beta within 2 SE at {n} rows): {verdict}")
        for pre in ("", "qml_") if "qml_alpha_diff_in_se" in rows[0] else ("",):
            bounds(rows, pre)
        if "nu_batch" in rows[0]:
            nus = np.array([r["nu_batch"] for r in rows])
            lo, mid, hi = nus.min(), np.median(nus), nus.max()
            print(f"  arch's nu: median {mid:.3g}, range {lo:.3g} to {hi:.3g}")


def bounds(rows: list[dict], pre: str) -> None:
    """The per-seed bounds the record's section 15.2 proposed for the targeted
    form: omega within 5 SE, alpha and beta within 3.5, path gap at most 5%."""

    def ok(r: dict, k: str, b: float) -> bool:
        return abs(r[f"{pre}{k}"]) <= b

    within2 = sum(ok(r, "alpha_diff_in_se", 2) and ok(r, "beta_diff_in_se", 2) for r in rows)
    om = sum(ok(r, "omega_diff_in_se", 5) for r in rows)
    ab = sum(ok(r, "alpha_diff_in_se", 3.5) and ok(r, "beta_diff_in_se", 3.5) for r in rows)
    path = sum(ok(r, "pred_median_rel_second_half", 0.05) for r in rows)
    every = sum(
        ok(r, "omega_diff_in_se", 5)
        and ok(r, "alpha_diff_in_se", 3.5)
        and ok(r, "beta_diff_in_se", 3.5)
        and ok(r, "pred_median_rel_second_half", 0.05)
        for r in rows
    )
    worst = max(rows, key=lambda r: max(abs(r[f"{pre}{p}_diff_in_se"]) for p in PARAMS))
    print(
        f"  [{pre or 'primary'}] within 2 SE {within2}; 15.2 bounds: omega <= 5 SE {om}, "
        f"alpha, beta <= 3.5 SE {ab}, path <= 5% {path}, all {every}; worst seed "
        f"{int(worst['seed'])} alpha {worst[pre + 'alpha_diff_in_se']:+.3g} "
        f"beta {worst[pre + 'beta_diff_in_se']:+.3g} SE"
    )


def main() -> None:
    if sys.argv[1:] == ["--check"]:
        check()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="+", type=Path)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--rows", nargs="+", type=int, default=list(SIZES))
    ap.add_argument("--persistence", type=float, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--nu", type=float, default=None)
    ap.add_argument("--remedy", default="", choices=["", *REMEDIES])
    a = ap.parse_args()
    if a.summary:
        summary(a.out)
        return
    i, k = (int(x) for x in a.shard.split("/"))
    run(a.out[0], tuple(a.rows), truth_at(a.persistence), (i, k), a.nu, a.remedy)


if __name__ == "__main__":
    main()
