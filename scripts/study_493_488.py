"""Study 493/488, request A17: the seasonal ETS of study 488 again, with the
admissible region enforced in the batch start, beta as a share of alpha, and
(part 2) a Huber score; and (parts 3 and 4) asymmetric DCC on the DCC engine
of study 485 against a batch ADCC written from the definition.

Part 1 (this commit) and part 2: ETS. Part 3 and 4: ADCC (added in later
commits). The record is ``docs/records/STUDY-493-488.md``.

numpy only in the estimators; statsmodels' ``ETSModel`` is the oracle (and, in
parts 3 and 4, scipy computes the ADCC likelihoods). Environment: a venv
``.venv-493`` in the repository root (excluded through ``.git/info/exclude``):

    uv venv .venv-493 && uv pip install --python .venv-493/bin/python numpy scipy statsmodels

Run::

    .venv-493/bin/python scripts/study_493_488.py part1    (ETS: region + beta = alpha b)
    .venv-493/bin/python scripts/study_493_488.py part2    (ETS: the Huber score; headroom)

``--seeds N`` runs the first N seeds only; ``--no-run`` summarises the cached
cases; ``--force`` recomputes a part's summary. Each case is appended to
``.cache/study_493_488/<part>.jsonl`` as it finishes, so a rerun resumes. The
baseline (study 488's batch start, the same 18 sets, the same seeds) is read
from ``.cache/study_487_488/{ets,ets_mul}.jsonl``, which
``scripts/study_487_488.py part3`` and ``part4`` write.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass

# one BLAS thread per worker, set before numpy loads (set afterwards it does
# nothing: four workers on four threads each ran at a load of 12)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "study_485"))
import study_487_488 as base  # noqa: E402

SEEDS = base.SEEDS
N_ROWS = base.N_ROWS
EPS = base.EPS
BASE_CACHE = os.path.join(".cache", "study_487_488")
OUT = os.path.join(".cache", "study_493_488")

# --------------------------------------------------------------------------
# the 18 sets of study 488: 12 additive (A, B truths) and 6 multiplicative (C, B)

ALL_SPECS: dict[str, base.EtsSpec] = {**base.ETS_SPECS, **base.MUL_SPECS}
SET_ORDER = tuple(base.P3_SPECS) + tuple(base.P4_SPECS)


@dataclass(frozen=True)
class Variant:
    """One estimator configuration. ``reparam``: beta = alpha b with b in
    [eps, 1 - eps], the step and the batch fit taken in (alpha, b, gamma[, phi]);
    ``adm``: the batch fit is held in the admissible region; ``huber``: the
    Huber constant (part 2), ``None`` for squared error."""

    name: str
    reparam: bool
    adm: bool
    huber: float | None = None


VARIANTS = {
    "rp": Variant("rp", True, False),
    "rp_adm": Variant("rp_adm", True, True),
    "rp_hub": Variant("rp_hub", True, False, 1.345),
    "rp_adm_hub": Variant("rp_adm_hub", True, True, 1.345),
}
PASS_COUNT = 45  # the criterion: at least 45 of 50 seeds
HEADROOM_COUNT = 47  # "with headroom": two seeds above the floor ...
HEADROOM_SE = 3.0  # ... and no seed over 3 SE on any parameter
PART_VARIANTS = {"part1": ("rp", "rp_adm"), "part2": ("rp_hub", "rp_adm_hub")}


# --------------------------------------------------------------------------
# the admissible region (Hyndman et al. 2008, section 10.6)


def theta_full(spec: base.EtsSpec, th: np.ndarray) -> tuple[float, float, float, float]:
    return (float(th[0]), float(th[1]), float(th[2]), float(th[3]) if spec.damped else 1.0)


def radius_minus_one(spec: base.EtsSpec, th: np.ndarray) -> float:
    """The largest eigenvalue modulus of the discount matrix ``D = F - g w'``
    after the one exact unit root, minus 1: negative inside the admissible
    region. For the multiplicative season this is the additive form's matrix at
    the same parameters, the linearisation about unit seasonal factors
    (``mul_region_check``)."""
    return base.ets_radius(theta_full(spec, th), spec.m) - 1.0


def admissible(spec: base.EtsSpec, th: np.ndarray) -> bool:
    return radius_minus_one(spec, th) < 0.0


# --------------------------------------------------------------------------
# the reparametrisation: theta' = (alpha, b, gamma[, phi]), beta = alpha b


def to_theta(spec: base.EtsSpec, tp: np.ndarray) -> np.ndarray:
    out = np.array(tp, dtype=float)
    out[1] = tp[0] * tp[1]
    return out


def to_prime(spec: base.EtsSpec, th: np.ndarray) -> np.ndarray:
    out = np.array(th, dtype=float)
    out[1] = th[1] / th[0]
    return out


def jacobian(spec: base.EtsSpec, tp: np.ndarray) -> np.ndarray:
    """``d theta / d theta'``: the identity but for row beta, ``(b, alpha)``."""
    J = np.eye(spec.n_params)
    J[1, 0], J[1, 1] = tp[1], tp[0]
    return J


def project_prime(spec: base.EtsSpec, tp: np.ndarray) -> np.ndarray:
    """statsmodels' box, written in the reparametrised coordinates: ``alpha``
    and ``b`` in ``[eps, 1 - eps]``, ``gamma`` in ``[eps (1 - alpha), (1 - eps)
    (1 - alpha)]``, ``phi`` in ``[0.8, 0.98]``."""
    out = np.array(tp, dtype=float)
    a = min(max(out[0], EPS), 1.0 - EPS)
    out[0] = a
    out[1] = min(max(out[1], EPS), 1.0 - EPS)
    out[2] = min(max(out[2], EPS * (1.0 - a)), (1.0 - EPS) * (1.0 - a))
    if spec.damped:
        out[3] = min(max(out[3], base.PHI_BOUNDS[0]), base.PHI_BOUNDS[1])
    return out


# --------------------------------------------------------------------------
# the batch start's objective and constrained fit


def huber_parts(e: float, c: float) -> tuple[float, float, float]:
    """Huber's rho, its derivative (the clipped residual) and its second
    derivative (1 inside the threshold, 0 outside)."""
    if abs(e) <= c:
        return 0.5 * e * e, e, 1.0
    return c * (abs(e) - 0.5 * c), c * math.copysign(1.0, e), 0.0


def score_objective(spec: base.EtsSpec, y: np.ndarray, huber: float | None):  # noqa: ANN201
    """``-sum rho(e_t)`` over ``y`` from the known initial states, its gradient
    ``sum rho'(e_t) psi_t`` and Gauss-Newton matrix ``sum rho''(e_t) psi_t
    psi_t'``, in theta. ``rho = e^2/2`` (``huber is None``, study 488's score) or
    Huber's with the fixed constant ``huber``."""
    if huber is None:
        return base.ets_objective(spec, y)

    def f(theta: np.ndarray, grad: bool):  # noqa: ANN202
        est = base.OnlineEts(spec)
        est.theta = np.array(theta, dtype=float)
        k = spec.n_params
        g = np.zeros(k)
        H = np.zeros((k, k))
        tot = 0.0
        for v in y:
            yhat = est.predict()
            e = float(v) - yhat
            rho, d1, d2 = huber_parts(e, huber)
            tot += rho
            if grad:
                psi = est._psi()
                g += d1 * psi
                H += d2 * np.outer(psi, psi)
            est._row(float(v), step=False)
            if spec.seasonal == "mul" and not (est.l + est.b > 0.0 and est.s.min() > 0.0):
                return -math.inf, g, H
        if not math.isfinite(tot):
            return -math.inf, g, H
        return -tot, g, H

    return f


class Objective:
    """The batch objective in theta' (value, gradient and Gauss-Newton matrix
    mapped through ``J = d theta / d theta'``). It does not look at the region:
    the constraint is held by ``maximize_region``'s steps."""

    def __init__(self, spec: base.EtsSpec, y: np.ndarray, var: Variant) -> None:
        self.spec, self.var = spec, var
        self.inner = score_objective(spec, y, var.huber)

    def __call__(self, tp: np.ndarray, grad: bool):  # noqa: ANN204
        sp = self.spec
        tp = np.asarray(tp, dtype=float)
        val, g, H = self.inner(to_theta(sp, tp), grad)
        if grad:
            J = jacobian(sp, tp)
            g, H = J.T @ g, J.T @ H @ J
        return val, g, H


MAX_ITER = 50
MAX_HALVINGS = 30
ARMIJO = 1e-4
TOL_X = 1e-8
TOL_F = 1e-10
RIDGE = 1e-10


def maximize_region(objective, spec: base.EtsSpec, grid: list[tuple]):  # noqa: ANN001, ANN201
    """``batch_core.maximize`` (projected Newton, observed curvature from
    central differences of the analytic gradient, Armijo backtracking, the same
    stopping rule) with the admissible region as the constraint
    ``r(theta') = radius(D) - 1 < 0`` held at every point the line search
    tries, as the projection ``Proj`` of a candidate ``x`` from the iterate
    ``theta``:

    1. the statsmodels box (``project_prime``);
    2. if ``r(x) >= 0``: the step to the linearised boundary in the metric of
       the curvature ``M``, ``x <- x - lam M^-1 grad r`` with ``lam = max(0, (r(theta)
       + grad r . (x - theta)) / (grad r' M^-1 grad r))``, which slides along the
       boundary instead of stopping on it (``grad r`` by forward differences of
       the radius, ``p`` evaluations, once per iteration and only when needed);
    3. if still ``r(x) >= 0``: bisection of the segment from ``theta`` to ``x``
       (8 halvings), ``theta`` itself being admissible.

    The start is the best admissible point of the grid. Returns a
    ``batch_core.Fit``; ``n_radius`` counts the radius evaluations."""
    import batch_core

    n_radius = 0

    def r(tp: np.ndarray) -> float:
        nonlocal n_radius
        n_radius += 1
        return radius_minus_one(spec, to_theta(spec, tp))

    evals = 0
    best, best_f = None, -math.inf
    for point in grid:
        tp = project_prime(spec, np.asarray(point, dtype=float))
        if r(tp) >= 0.0:
            continue
        fv, _, _ = objective(tp, False)
        evals += 1
        if math.isfinite(fv) and fv > best_f:
            best, best_f = tp, fv
    assert best is not None, "no admissible start"
    theta = best
    f, g, H = objective(theta, True)
    evals += 1
    r0 = r(theta)
    stop, it, newton = "iteration cap", 0, 0
    while it < MAX_ITER:
        it += 1
        p = H.shape[0]
        M = batch_core._curvature(objective, theta, p)
        evals += 2 * p
        if M is None:
            M = H
        else:
            newton += 1
        ridge = RIDGE * max(float(np.trace(M)) / p, 1e-300)
        Mr = M + ridge * np.eye(p)
        d = np.linalg.solve(Mr, g)
        cache: dict = {}

        def grad_r(theta=theta, cache=cache) -> np.ndarray:  # noqa: ANN001
            if "g" not in cache:
                gr = np.empty(p)
                for j in range(p):
                    h = 1e-6 * max(abs(float(theta[j])), 1e-2)
                    e = np.zeros(p)
                    e[j] = h
                    gr[j] = (r(theta + e) - r0) / h
                cache["g"] = gr
            return cache["g"]

        def proj(x: np.ndarray, theta=theta, Mr=Mr, r0=r0) -> np.ndarray:  # noqa: ANN001
            x = project_prime(spec, x)
            if r(x) < 0.0:
                return x
            gr = grad_r()
            Mg = np.linalg.solve(Mr, gr)
            lam = max(0.0, (r0 + float(gr @ (x - theta))) / max(float(gr @ Mg), 1e-300))
            x2 = project_prime(spec, x - lam * Mg)
            if r(x2) < 0.0:
                return x2
            lo, hi = 0.0, 1.0
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                if r(theta + mid * (x - theta)) < 0.0:
                    lo = mid
                else:
                    hi = mid
            return theta + lo * (x - theta)

        accepted = None
        for direction in batch_core._directions(d, g, M, ridge, theta, project_prime_fn(spec)):
            s = 1.0
            for _ in range(MAX_HALVINGS + 1):
                cand = proj(theta + s * direction)
                fc, _, _ = objective(cand, False)
                evals += 1
                if math.isfinite(fc) and fc >= f + ARMIJO * float(g @ (cand - theta)):
                    accepted = (cand, fc)
                    break
                s *= 0.5
            if accepted is not None:
                break
        if accepted is None:
            stop = "no ascent"
            break
        cand, fc = accepted
        moved = float(np.max(np.abs(cand - theta)))
        df = abs(fc - f)
        theta = cand
        f, g, H = objective(theta, True)
        evals += 1
        r0 = r(theta)
        if moved <= TOL_X and df <= TOL_F * (1.0 + abs(f)):
            stop = "converged"
            break
    fit = batch_core.Fit(
        theta=theta, value=f, iterations=it, stop=stop, evaluations=evals, newton_steps=newton
    )
    fit.n_radius = n_radius  # type: ignore[attr-defined]
    return fit


def project_prime_fn(spec: base.EtsSpec):  # noqa: ANN201
    return lambda x: project_prime(spec, x)


def start_grid(spec: base.EtsSpec, var: Variant) -> list[tuple]:
    """Study 488's start grid, in theta' (``b`` as ``beta/alpha``, so the same
    points), and for the constrained fit two points deep inside the region
    (alpha 0.01 and 0.05, b 0.01 and 0.05, gamma/(1 - alpha) 0.001), where the
    region of a long season still has room."""
    pts: list[tuple] = []
    base_pts = [
        (a, bs, (1.0 - a) * gs)
        for a, bs, gs in itertools.product((0.1, 0.3, 0.6), (0.05, 0.3), (0.05, 0.3))
    ]
    if var.adm:
        base_pts += [(0.01, 0.01, 0.99 * 0.001), (0.05, 0.05, 0.95 * 0.001)]
    for p in base_pts:
        pts += [p + (phi,) for phi in (0.85, 0.95)] if spec.damped else [p]
    return pts


def fit_start(spec: base.EtsSpec, y: np.ndarray, var: Variant):  # noqa: ANN201
    """The batch start's fit in theta': ``batch_core.maximize`` unchanged (the
    box only), or ``maximize_region`` (the box and the admissible region)."""
    import batch_core

    obj = Objective(spec, y, var)
    if var.adm:
        return maximize_region(obj, spec, start_grid(spec, var))
    fit = batch_core.maximize(obj, lambda x: project_prime(spec, x), start_grid(spec, var))
    fit.n_radius = 0  # type: ignore[attr-defined]
    return fit


# --------------------------------------------------------------------------
# the online estimator


class OnlineEtsRp(base.OnlineEts):
    """Study 488's ``OnlineEts`` with the step taken in theta' and the batch
    start of ``fit_start``. The sensitivities stay in theta; at each step the
    gradient and the information are mapped, ``g' = J' g`` and ``R' = J' R J``
    with ``J = d theta / d theta'`` at the current parameters, and
    ``theta' <- Proj(theta' + R'^-1 g' / W)``. State: the same floats as
    ``OnlineEts`` (``theta`` is stored; ``b = beta/alpha`` is derived)."""

    def __init__(self, spec: base.EtsSpec, var: Variant, batch_start: int = 0) -> None:
        super().__init__(spec, batch_start=batch_start)
        self.var = var
        self.fit_radius = math.nan
        self.radii: list[tuple[int, float]] = []

    def _start(self) -> None:
        rows = np.asarray(self.buffer)
        self.batch_fit = fit_start(self.spec, rows, self.var)
        tp = project_prime(self.spec, self.batch_fit.theta)
        self.theta = to_theta(self.spec, tp)
        self.fit_radius = radius_minus_one(self.spec, self.theta)
        for v in rows:
            self._row(float(v), step=False, replay=True)
        self.buffer = []
        self.started = True

    def _row(self, y: float, step: bool, replay: bool = False) -> None:
        if not step:
            return super()._row(y, step, replay)
        psi = self._psi()
        e = y - self.predict()
        super()._row(y, False, replay)  # states, sensitivities, R, s2, W
        if not self.batch_start and self.t <= self.warm:
            return
        sp = self.spec
        th = self.theta
        tp = to_prime(sp, th)
        J = jacobian(sp, tp)
        d1 = e
        if self.var.huber is not None:
            d1 = huber_parts(e, self.var.huber)[1]
        gp = J.T @ (psi * d1)
        Rp = J.T @ self.R @ J
        stepv = (1.0 / self.W) * np.linalg.solve(Rp, gp)
        self.theta = to_theta(sp, project_prime(sp, tp + stepv))


# --------------------------------------------------------------------------
# one case: the oracle fit and each variant on the same series


def run_variant(spec: base.EtsSpec, var: Variant, y: np.ndarray) -> dict:
    n = y.shape[0]
    t0 = time.perf_counter()
    est = OnlineEtsRp(spec, var, batch_start=base.ets_window(spec))
    pred = np.empty(n)
    radii = []
    for t, v in enumerate(y):
        pred[t] = est.update(float(v))
        if (t + 1) % 2000 == 0:
            radii.append(radius_minus_one(spec, est.theta))
    return {
        "est": est,
        "pred": pred,
        "seconds": time.perf_counter() - t0,
        "radii": radii,
    }


def case(spec_name: str, seed: int, variants: tuple[str, ...], n: int = N_ROWS) -> dict:
    spec = ALL_SPECS[spec_name]
    y = base.simulate_ets(spec, n, seed)
    t0 = time.perf_counter()
    model, res = base.etsmodel_fit(spec, y)
    t_batch = time.perf_counter() - t0
    b = np.concatenate([np.asarray(res.params, dtype=float), [float(res.mse)]])
    se = np.concatenate([np.asarray(res.bse, dtype=float), [float(res.mse) * math.sqrt(2.0 / n)]])
    fitted = np.asarray(res.fittedvalues, dtype=float)
    half = slice(n // 2, n)
    out: dict = {
        "spec": spec_name,
        "seed": seed,
        "n": n,
        "sm_converged": bool(res.mle_retvals.get("converged", False)),
        "oracle_radius_minus_1": radius_minus_one(spec, b[:-1]),
        "t_batch_s": t_batch,
    }
    for i, nm in enumerate(spec.names):
        out[f"{nm}_batch"] = float(b[i])
        out[f"{nm}_se"] = float(se[i])
    for vname in variants:
        var = VARIANTS[vname]
        r = run_variant(spec, var, y)
        est, pred = r["est"], r["pred"]
        o = est.params
        pre = vname + "."
        yh = pred[half]
        out[pre + "silent_rows"] = int(np.sum(np.isnan(pred)))
        out[pre + "silent_explained"] = int(
            sum(1 for i in range(n) if math.isnan(pred[i]) and i in est.reasons)
        )
        out[pre + "llf_online_params"] = float(model.loglike(o[:-1]))
        out[pre + "pred_rms_gap_second_half"] = float(np.sqrt(np.mean((yh - fitted[half]) ** 2)))
        out[pre + "online_mse_second_half"] = float(np.mean((y[half] - yh) ** 2))
        out[pre + "batch_mse_second_half"] = float(np.mean((y[half] - fitted[half]) ** 2))
        out[pre + "t_online_s"] = r["seconds"]
        out[pre + "state_bytes"] = est.state_bytes()
        out[pre + "start_fit_stop"] = est.batch_fit.stop
        out[pre + "start_fit_iterations"] = est.batch_fit.iterations
        out[pre + "start_fit_evaluations"] = est.batch_fit.evaluations
        out[pre + "start_fit_radius_evals"] = est.batch_fit.n_radius
        out[pre + "start_fit_radius_minus_1"] = est.fit_radius
        out[pre + "radius_minus_1_by_2000_rows"] = r["radii"]
        for i, nm in enumerate(spec.names):
            out[f"{pre}{nm}_online"] = float(o[i])
            out[f"{pre}{nm}_gap_se"] = float((o[i] - b[i]) / se[i])
        gaps = [out[f"{pre}{nm}_gap_se"] for nm in spec.names]
        out[pre + "max_abs_gap_se"] = float(
            max((abs(x) for x in gaps if not math.isnan(x)), default=math.nan)
        )
    return out


def _case_args(a):  # noqa: ANN001, ANN202
    return case(*a)


def run_cases(path: str, cases: list[tuple], workers: int) -> list[dict]:
    done = {(r["spec"], r["seed"]) for r in base._load(path)}
    todo = [c for c in cases if (c[0], c[1]) not in done]
    if todo:
        print(f"{path}: {len(done)} cached, {len(todo)} to run", flush=True)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from multiprocessing import get_context

        with get_context("fork").Pool(workers) as pool, open(path, "a") as fh:
            for i, res in enumerate(pool.imap_unordered(_case_args, todo)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 25 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
    return base._load(path)


# --------------------------------------------------------------------------
# counts


def baseline_rows() -> dict[str, list[dict]]:
    """Study 488's batch-start rows by set, read from its cache, with the gaps
    recomputed from the stored parameters (``batch.<name>_online``)."""
    rows: dict[str, list[dict]] = {}
    for fn in ("ets.jsonl", "ets_mul.jsonl"):
        for r in base._load(os.path.join(BASE_CACHE, fn)):
            rows.setdefault(r["spec"], []).append(r)
    return rows


def gaps_of(r: dict, spec: base.EtsSpec, pre: str) -> list[float]:
    return [r[f"{pre}{nm}_gap_se"] for nm in spec.names]


def count_within(rows: list[dict], spec: base.EtsSpec, pre: str, only=None) -> dict:  # noqa: ANN001
    """Seeds with every parameter within 2 of ``ETSModel``'s SEs. A parameter
    whose SE is NaN (``ETSModel`` gives none at a bound) is not compared and
    the seed is counted on the others, as study 488 counted it; ``nan_se``
    counts those seeds. ``only(row)`` restricts the seeds."""
    rows = [r for r in rows if pre + "smoothing_level_gap_se" in r and (only is None or only(r))]
    n_in, nan_se, worst = 0, 0, (0.0, None, None)
    per_param: dict[str, list[float]] = {nm: [] for nm in spec.names}
    for r in rows:
        gs = gaps_of(r, spec, pre)
        if any(math.isnan(x) for x in gs):
            nan_se += 1
        ok = [abs(x) for x in gs if not math.isnan(x)]
        if ok and max(ok) <= 2.0:
            n_in += 1
        for nm, x in zip(spec.names, gs, strict=True):
            if not math.isnan(x):
                per_param[nm].append(abs(x))
        for nm, x in zip(spec.names, gs, strict=True):
            if not math.isnan(x) and abs(x) > worst[0]:
                worst = (abs(x), nm, r["seed"])
    out = {
        "cases": len(rows),
        "within2": n_in,
        "nan_se": nan_se,
        "worst_gap_se": worst[0],
        "worst_param": worst[1],
        "worst_seed": worst[2],
        "median_gap_se": {nm: float(np.median(v)) if v else math.nan for nm, v in per_param.items()},
    }
    return out


def summarise_set(
    name: str, rows_new: list[dict], rows_base: list[dict], variants: tuple[str, ...]
) -> dict:
    spec = ALL_SPECS[name]
    out: dict = {"spec": name, "label": spec.label, "truth": spec.truth[:-1]}
    out["baseline"] = count_within(rows_base, spec, "batch.")
    # a baseline restricted to the same seeds as the new rows
    seeds = {r["seed"] for r in rows_new}
    sub = [r for r in rows_base if r["seed"] in seeds]
    out["baseline_same_seeds"] = count_within(sub, spec, "batch.")
    rad = np.array([r["oracle_radius_minus_1"] for r in rows_new]) if rows_new else np.array([])
    out["oracle_outside_region"] = int(np.sum(rad > 0.0))
    out["oracle_radius_minus_1_median"] = float(np.median(rad)) if rad.size else math.nan
    truth = tuple(spec.truth[:-1])
    out["truth_radius_minus_1"] = radius_minus_one(spec, np.array(truth))
    for v in variants:
        pre = v + "."
        d = count_within(rows_new, spec, pre)
        din = count_within(
            rows_new, spec, pre, only=lambda r: r["oracle_radius_minus_1"] <= 0.0
        )
        d["oracle_inside_region"] = din
        d["passes"] = d["within2"] >= PASS_COUNT
        d["headroom"] = d["within2"] >= HEADROOM_COUNT and d["worst_gap_se"] <= HEADROOM_SE
        rs = [r for r in rows_new if pre + "max_abs_gap_se" in r]
        if rs:
            d["start_fit_converged"] = int(sum(r[pre + "start_fit_stop"] == "converged" for r in rs))
            d["start_fit_stops"] = {
                s: int(sum(r[pre + "start_fit_stop"] == s for r in rs))
                for s in ("converged", "no ascent", "iteration cap")
            }
            fr = np.array([r[pre + "start_fit_radius_minus_1"] for r in rs])
            d["start_fit_outside_region"] = int(np.sum(fr > 0.0))
            end = np.array([r[pre + "radius_minus_1_by_2000_rows"][-1] for r in rs])
            d["final_online_outside_region"] = int(np.sum(end > 0.0))
            d["pred_rms_gap_median"] = float(np.median([r[pre + "pred_rms_gap_second_half"] for r in rs]))
            d["mse_ratio_median"] = float(
                np.median([r[pre + "online_mse_second_half"] / r[pre + "batch_mse_second_half"] for r in rs])
            )
            d["mse_ratio_max"] = float(
                np.max([r[pre + "online_mse_second_half"] / r[pre + "batch_mse_second_half"] for r in rs])
            )
            d["t_online_median_s"] = float(np.median([r[pre + "t_online_s"] for r in rs]))
            d["state_bytes"] = rs[0][pre + "state_bytes"]
            d["silent_unexplained"] = int(sum(r[pre + "silent_rows"] - r[pre + "silent_explained"] for r in rs))
        out[v] = d
    return out


def print_summary(summ: list[dict], variants: tuple[str, ...]) -> None:
    hdr = f"{'set':14s} {'truth r-1':>9s} {'oracle out':>10s} {'base':>5s} " + " ".join(
        f"{v + ' (in-region oracle)':>24s}" for v in variants
    )
    print(hdr)
    for s in summ:
        cells = []
        for v in variants:
            d = s[v]
            din = d["oracle_inside_region"]
            cells.append(f"{d['within2']:3d} of {d['cases']:2d} ({din['within2']:2d}/{din['cases']:2d})".rjust(24))
        print(
            f"{s['spec']:14s} {s['truth_radius_minus_1']:+9.1e} {s['oracle_outside_region']:6d}/50 "
            f"{s['baseline']['within2']:5d} " + " ".join(cells)
        )
    print()
    for v in ("baseline",) + variants:
        if v == "baseline":
            ps = [x["spec"] for x in summ if x["baseline"]["within2"] >= PASS_COUNT]
            print(f"passes (>= {PASS_COUNT}) {v:11s}: {len(ps)} sets: {', '.join(ps)}")
            continue
        ps = [x["spec"] for x in summ if x[v]["passes"]]
        hs = [x["spec"] for x in summ if x[v]["headroom"]]
        print(f"passes (>= {PASS_COUNT}) {v:11s}: {len(ps)} sets: {', '.join(ps)}")
        print(f"  with headroom (>= {HEADROOM_COUNT}, worst <= {HEADROOM_SE} SE): {len(hs)}: {', '.join(hs)}")
    for s in summ:
        print(f"-- {s['spec']} ({s['label']}), truth {s['truth']}")
        b = s["baseline"]
        print(
            f"   baseline (488): {b['within2']}/{b['cases']}, worst {b['worst_gap_se']:.3g} SE "
            f"({b['worst_param']}, seed {b['worst_seed']}), nan-SE seeds {b['nan_se']}"
        )
        for v in variants:
            d = s[v]
            if not d["cases"]:
                continue
            print(
                f"   {v}: {d['within2']}/{d['cases']}, worst {d['worst_gap_se']:.3g} SE "
                f"({d['worst_param']}, seed {d['worst_seed']}); start fits {d['start_fit_stops']}; "
                f"fit outside region {d['start_fit_outside_region']}, final online outside "
                f"{d['final_online_outside_region']}; RMS gap {d['pred_rms_gap_median']:.3g}; "
                f"MSE ratio median {d['mse_ratio_median']:.4f} max {d['mse_ratio_max']:.4f}; "
                f"{d['t_online_median_s']:.2f} s/case; silent unexplained {d['silent_unexplained']}"
            )


# --------------------------------------------------------------------------
# checks


def mul_region_check(m: int = 24, th=(0.3, 0.001, 0.05, 1.0)) -> dict:  # noqa: ANN001
    """The multiplicative form's region against the additive one's. Along the
    nominal orbit (no errors), the monodromy over one season of the
    multiplicative filter's one-step map, by finite differences, has the
    eigenvalues whose ``m``-th roots are the per-row growth; compare the largest
    after the unit root with the additive ``D``'s over the same season
    (``radius ** m``)."""
    spec = base.EtsSpec("chk", False, m, th + (1.0,), "mul")
    est0 = base.OnlineEts(spec)
    est0.theta = np.array(th[:3], dtype=float)
    n = m + 2

    def pack(e: base.OnlineEts) -> np.ndarray:
        s = np.roll(e.s, -e.pos)
        return np.concatenate([[e.l, e.b], s])

    def run(x: np.ndarray, orbit: list[float] | None) -> tuple[np.ndarray, list[float]]:
        e = base.OnlineEts(spec)
        e.theta = np.array(th[:3], dtype=float)
        e.l, e.b = x[0], x[1]
        e.s = x[2:].copy()
        e.pos = 0
        ys: list[float] = []
        for t in range(m):
            yt = e.predict() if orbit is None else orbit[t]
            ys.append(yt)
            e._row(yt, step=False)
        return pack(e), ys

    x0 = pack(est0)
    x1, orbit = run(x0, None)
    h = 1e-6
    Mono = np.empty((n, n))
    for j in range(n):
        xp = x0.copy()
        xp[j] += h
        xm = x0.copy()
        xm[j] -= h
        Mono[:, j] = (run(xp, orbit)[0] - run(xm, orbit)[0]) / (2 * h)
    ev = np.linalg.eigvals(Mono)
    ev = np.delete(ev, int(np.argmin(np.abs(ev - 1.0))))
    mul_rad = float(np.max(np.abs(ev)))
    add_rad = base.ets_radius(th, m) ** m
    return {
        "m": m,
        "theta": th[:3],
        "multiplicative_season_monodromy_radius": mul_rad,
        "additive_D_radius_pow_m": add_rad,
        "per_row_multiplicative": mul_rad ** (1.0 / m),
        "per_row_additive": base.ets_radius(th, m),
    }


def radius_poly_check() -> list[dict]:
    """The radius of ``D`` against an independent route: the moving-average
    polynomial of the innovations form (undamped), ``(1 - B)^2 (1 - B^m) y =
    Theta(B) e`` with ``Theta = (1 - B)^2 (1 - B^m) + B (alpha + beta - alpha B)
    (1 - B^m) + gamma B^m (1 - B)^2``; its roots' reciprocals are ``D``'s
    eigenvalues (one is exactly 1)."""
    from numpy.polynomial import polynomial as P

    def rad(a: float, beta: float, g: float, m: int) -> float:
        omb = np.array([1.0, -1.0])
        omb2 = P.polymul(omb, omb)
        ombm = np.zeros(m + 1)
        ombm[0], ombm[m] = 1.0, -1.0
        gm = np.zeros(m + 1)
        gm[m] = g
        th = P.polyadd(
            P.polyadd(P.polymul(omb2, ombm), P.polymul(np.array([0.0, a + beta, -a]), ombm)),
            P.polymul(gm, omb2),
        )
        z = 1.0 / np.roots(th[::-1])
        z = np.delete(z, int(np.argmin(np.abs(z - 1.0))))
        return float(np.max(np.abs(z)))

    out = []
    for m in (7, 24, 168):
        for a, bs, gs in ((0.3, 0.01, 0.001), (0.3, 0.1 / 0.3, 0.1 / 0.7), (0.1, 0.01, 0.001)):
            beta, g = a * bs, (1.0 - a) * gs
            r1 = base.ets_radius((a, beta, g, 1.0), m) - 1.0
            r2 = rad(a, beta, g, m) - 1.0
            out.append({"m": m, "alpha": a, "beta": beta, "gamma": g, "matrix": r1, "polynomial": r2})
    return out


def checks() -> dict:
    """Held on one series of 3,000 rows per spec: the carried gradient in theta'
    against a central difference of the objective (squared and Huber), the
    constrained fit ends inside the region, out of sample (row 1,500 changed by
    100), allocated floats equal the state count."""
    out: dict = {"radius_poly": radius_poly_check(), "mul_region": [mul_region_check(24), mul_region_check(7)]}
    rows = []
    for name in ("aada_m24_b", "aaa_m7"):
        spec = ALL_SPECS[name]
        y = base.simulate_ets(spec, 3000, 493)
        for huber in (None, 1.345):
            var = Variant("chk", True, False, huber)
            c = Objective(spec, y, var)
            tp = to_prime(spec, np.array(spec.truth[:-1]) * 0.9)
            _, g, _ = c(tp, True)
            num = np.zeros_like(g)
            for j in range(spec.n_params):
                h = 1e-6
                tpp, tpm = tp.copy(), tp.copy()
                tpp[j] += h
                tpm[j] -= h
                num[j] = (c(tpp, False)[0] - c(tpm, False)[0]) / (2 * h)
            rows.append(
                {
                    "spec": name,
                    "huber": huber,
                    "gradient_rel_err_theta_prime": float(np.max(np.abs(g - num)) / np.max(np.abs(num))),
                }
            )
    out["gradient"] = rows
    spec = ALL_SPECS["aada_m24_b"]
    y = base.simulate_ets(spec, 3000, 493)
    var = VARIANTS["rp_adm"]
    a = OnlineEtsRp(spec, var, batch_start=base.ets_window(spec))
    b2 = OnlineEtsRp(spec, var, batch_start=base.ets_window(spec))
    moved = 0.0
    for t, v in enumerate(y):
        pa = a.update(float(v))
        pb = b2.update(float(v) + (100.0 if t == 1500 else 0.0))
        if t <= 1500 and not (math.isnan(pa) or math.isnan(pb)):
            moved = max(moved, abs(pa - pb))
    out["out_of_sample_max_move_up_to_changed_row"] = moved
    out["allocated_floats"] = a.allocated_floats()
    out["state_floats"] = sum(base.ets_state_floats(spec).values())
    out["fit_in_region"] = a.fit_radius < 0.0
    return out


# --------------------------------------------------------------------------
# parts


# --------------------------------------------------------------------------
# the state caps (the user's rule: 256 KB per group by default, warn at half,
# refuse at the cap; a bank-wide total that can be infinite)

CAP_BYTES = 256 * 1024
WARN_BYTES = CAP_BYTES // 2
CAP_MS = (7, 12, 24, 52, 96, 168, 336, 720, 1440, 2184, 4320, 8760)


def cap_verdict(nbytes: int) -> str:
    return "refuse" if nbytes >= CAP_BYTES else ("warn" if nbytes >= WARN_BYTES else "ok")


def ets_bytes(k: int, m: int) -> int:
    """The estimator's state: level, trend, the season ring, their
    sensitivities to ``k`` parameters (``(2 + m) k``), ``theta``, ``R``
    (``k (k + 1) / 2``) and four scalars: ``8 ((k + 1) (2 + m) + k +
    k (k + 1) / 2 + 4)``, i.e. ``32 m + 168`` (k = 3) and ``40 m + 224`` (k = 4)."""
    return 8 * ((k + 1) * (2 + m) + k + k * (k + 1) // 2 + 4)


def ets_buffer_bytes(m: int) -> int:
    """The batch start's buffer, held until the start is fitted."""
    return 8 * max(500, 3 * m)


def largest_m(fn, limit: int) -> int:  # noqa: ANN001
    """The largest ``m`` with ``fn(m) < limit`` (``fn`` increasing)."""
    lo, hi = 1, 10**7
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fn(mid) < limit:
            lo = mid
        else:
            hi = mid - 1
    return lo


def state_caps() -> dict:
    out: dict = {"cap_bytes": CAP_BYTES, "warn_bytes": WARN_BYTES, "rows": [], "limits": {}}
    for m in CAP_MS:
        row: dict = {"m": m, "buffer": ets_buffer_bytes(m), "region_matrix": 8 * (m + 2) ** 2}
        for k, nm in ((3, "aaa"), (4, "aada")):
            b = ets_bytes(k, m)
            row[nm] = b
            row[nm + "_peak"] = b + ets_buffer_bytes(m)
            row[nm + "_verdict"] = cap_verdict(b)
            row[nm + "_peak_verdict"] = cap_verdict(b + ets_buffer_bytes(m))
        out["rows"].append(row)
    for k, nm in ((3, "aaa"), (4, "aada")):
        out["limits"][nm] = {
            "state_warn_m": largest_m(lambda m, k=k: ets_bytes(k, m), WARN_BYTES) + 1,
            "state_refuse_m": largest_m(lambda m, k=k: ets_bytes(k, m), CAP_BYTES) + 1,
            "peak_warn_m": largest_m(lambda m, k=k: ets_bytes(k, m) + ets_buffer_bytes(m), WARN_BYTES) + 1,
            "peak_refuse_m": largest_m(lambda m, k=k: ets_bytes(k, m) + ets_buffer_bytes(m), CAP_BYTES) + 1,
        }
    out["limits"]["region_matrix_over_cap_m"] = largest_m(lambda m: 8 * (m + 2) ** 2, CAP_BYTES) + 1
    # the formulas against the floats the estimator allocates, per m and form
    chk = []
    for m in (7, 24, 168, 1440):
        for damped in (False, True):
            spec = base.EtsSpec("cap", damped, m, (0.3, 0.01, 0.05) + ((0.9,) if damped else ()) + (1.0,))
            est = OnlineEtsRp(spec, VARIANTS["rp_adm_hub"], batch_start=base.ets_window(spec))
            chk.append(
                {
                    "m": m,
                    "damped": damped,
                    "allocated_bytes": 8 * est.allocated_floats(),
                    "formula_bytes": ets_bytes(spec.n_params, m),
                    "state_bytes_method": est.state_bytes(),
                }
            )
    out["allocation_check"] = chk
    out["bank_total_note"] = "bank total = sum over groups of the per-group bytes; with the total cap at infinity only the per-group cap binds"
    return out


def ets_part(part: str, args) -> dict:  # noqa: ANN001
    variants = PART_VARIANTS[part]
    path = os.path.join(args.out, f"{part}.jsonl")
    seeds = SEEDS[: args.seeds]
    cases = [(k, s, variants) for k in SET_ORDER for s in seeds]
    rows = base._load(path) if args.no_run else run_cases(path, cases, args.workers)
    by_set: dict[str, list[dict]] = {}
    for r in rows:
        by_set.setdefault(r["spec"], []).append(r)
    bl = baseline_rows()
    res: dict = {"seeds": [seeds[0], seeds[-1]], "n": N_ROWS, "variants": list(variants)}
    if part == "part1":
        res["checks"] = checks()
    if part == "part2":
        res["state_caps"] = state_caps()
        res["checks"] = {"gradient": checks()["gradient"]}
    res["summary"] = [
        summarise_set(k, sorted(by_set.get(k, []), key=lambda r: r["seed"]), bl.get(k, []), variants)
        for k in SET_ORDER
    ]
    have = {(r["spec"], r["seed"]) for r in rows}
    res["complete"] = all((c[0], c[1]) in have for c in cases)
    return res


def summarise_ets(res: dict) -> None:
    if "checks" in res:
        for k, v in res["checks"].items():
            print("check", k, v)
    if "state_caps" in res:
        sc = res["state_caps"]
        print(f"cap {sc['cap_bytes']} B, warn {sc['warn_bytes']} B; limits {sc['limits']}")
        print("m, aaa, aada, buffer, aaa+buffer, aada+buffer, region matrix; verdicts")
        for r in sc["rows"]:
            print(
                r["m"], r["aaa"], r["aada"], r["buffer"], r["aaa_peak"], r["aada_peak"], r["region_matrix"],
                r["aaa_verdict"], r["aada_verdict"], r["aaa_peak_verdict"], r["aada_peak_verdict"],
            )
        for c in sc["allocation_check"]:
            print("allocation", c)
    print("complete:", res["complete"])
    print_summary(res["summary"], tuple(res["variants"]))


def versions() -> dict:
    return base.versions()


PARTS = {
    "part1": (lambda a: ets_part("part1", a), summarise_ets),
    "part2": (lambda a: ets_part("part2", a), summarise_ets),
}


def main(argv=None) -> int:  # noqa: ANN001
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(PARTS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="the first N seeds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="recompute the part's summary")
    ap.add_argument("--no-run", action="store_true", help="summarise the cached cases; run nothing")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.part}.json")
    run, summarise = PARTS[args.part]
    if os.path.exists(path) and not args.force:
        with open(path) as fh:
            res = json.load(fh)
        print(f"(cached: {path})")
    else:
        t0 = time.perf_counter()
        res = run(args)
        res["versions"] = versions()
        res["_seconds"] = time.perf_counter() - t0
        if res.get("complete", True):
            with open(path, "w") as fh:
                json.dump(res, fh, indent=1)
    summarise(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
