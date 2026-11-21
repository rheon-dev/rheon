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
from dataclasses import dataclass as dataclasses_dataclass
from dataclasses import field as dataclasses_field

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


# --------------------------------------------------------------------------
# asymmetric DCC (Cappiello, Engle and Sheppard 2006) on study 485's DCC engine
#
#   Q_t = Qbar - a Qbar - b Qbar - g N + a z_{t-1} z_{t-1}' + g n_{t-1} n_{t-1}' + b Q_{t-1}
#   n_t = z_t * (z_t < 0) elementwise,   N = mean of n n',   Qbar = mean of z z'
#   R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)
#
# The admissible set is a, b, g >= 0 and a + b + delta g < 1, delta the largest
# generalized eigenvalue of N against Qbar (the largest eigenvalue of
# Qbar^(-1/2) N Qbar^(-1/2)): it makes the intercept C = (1 - a - b) Qbar - g N
# positive semi-definite, so every Q_t is (CES 2006, section 3). The engine's
# second stage is carried over unchanged: the margins' targeted online GARCH
# (batch start of 500 rows, Huber score 2), the shared start (both stages
# fitted on the margins' 500 rows by the core's optimizer), a Student-t second
# stage with nu fitted at the start and held, Fisher scoring at gain
# 1/(t + 500), information averaged at the same gain, Qbar and N running means.

import dcc  # noqa: E402  (study_485/dcc.py, carried byte for byte)
import garch  # noqa: E402  (study_485/garch.py)

ADCC_TRUTH = (0.03, 0.93, 0.03)  # a, b, g
ADCC_NAMES = ("a", "b", "g")
ADCC_RHO_BAR = 0.5
ADCC_NU = 5.0
ADCC_EPS = 1e-4
ADCC_ROWS = 10_000
ADCC_START = 500
ADCC_GRID_A_B = dcc_grid = [(a, b) for a in (0.01, 0.03, 0.05, 0.1) for b in (0.8, 0.85, 0.9, 0.95)]


def neg_part(z: np.ndarray) -> np.ndarray:
    """``n = z * (z < 0)`` elementwise."""
    return np.minimum(z, 0.0)


def n_mean(z: np.ndarray) -> np.ndarray:
    nv = neg_part(z)
    return nv.T @ nv / z.shape[0]


def delta_of(qbar: np.ndarray, nmat: np.ndarray) -> float:
    """The largest eigenvalue of ``Qbar^(-1/2) N Qbar^(-1/2)``."""
    L = np.linalg.cholesky(qbar)
    Li = np.linalg.inv(L)
    return float(np.linalg.eigvalsh(Li @ nmat @ Li.T)[-1])


def project_adcc(x: np.ndarray, delta: float) -> np.ndarray:
    """Euclidean projection of ``(a, b, g)`` onto ``a, b, g >= 0`` and
    ``a + b + delta g <= 1 - 1e-4``: clip, and if the sum is over, the
    Lagrange form ``theta_i = max(0, x_i - lam c_i)`` with ``c = (1, 1,
    delta)`` and ``lam`` by bisection so that ``c . theta`` meets the cap
    (the KKT solution of the projection)."""
    cap = 1.0 - ADCC_EPS
    c = np.array([1.0, 1.0, delta])
    x = np.asarray(x, dtype=float)
    y = np.maximum(x, 0.0)
    if float(c @ y) <= cap:
        return y
    lo, hi = 0.0, float(np.max(x / c))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(c @ np.maximum(x - mid * c, 0.0)) > cap:
            lo = mid
        else:
            hi = mid
    return np.maximum(x - hi * c, 0.0)


def simulate_adcc(
    n: int,
    seed: int,
    k: int,
    abg: tuple[float, float, float],
    rho_bar: float,
    n0: np.ndarray,
    nu: float | None = None,
    burn: int = 1000,
    margin: tuple[float, float, float] = dcc.MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` rows of ``k`` returns (GARCH(1,1) margins) and their standardized
    residuals ``z``: ADCC correlation with the intercept ``(1 - a - b) Qbar - g N0``
    (``N0`` the model's own ``E[n n']``, ``calibrate_n0``), Gaussian or
    multivariate Student-t (unit variances) innovations, as ``dcc.simulate_k``."""
    a, b, g = abg
    w, al, be = margin
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n + burn, k))
    scale = np.ones(n + burn) if nu is None else np.sqrt((nu - 2.0) / rng.chisquare(nu, n + burn))
    qbar = np.full((k, k), rho_bar)
    np.fill_diagonal(qbar, 1.0)
    c0 = (1.0 - a - b) * qbar - g * n0
    assert np.linalg.eigvalsh(c0)[0] >= 0.0, "the truth's intercept is not positive semi-definite"
    q = qbar.copy()
    s2 = np.full(k, w / (1.0 - al - be))
    prev_r2 = s2.copy()
    prev_z = np.zeros(k)
    out = np.empty((n + burn, k))
    zs = np.empty((n + burn, k))
    for t in range(n + burn):
        if t > 0:
            nv = np.minimum(prev_z, 0.0)
            q = c0 + a * np.outer(prev_z, prev_z) + g * np.outer(nv, nv) + b * q
        d = 1.0 / np.sqrt(np.diag(q))
        z = np.linalg.cholesky(q * np.outer(d, d)) @ x[t] * scale[t]
        s2 = w + al * prev_r2 + be * s2
        out[t] = np.sqrt(s2) * z
        zs[t] = z
        prev_r2 = out[t] ** 2
        prev_z = z
    return out[burn:], zs[burn:]


def calibrate_n0(k: int, nu: float | None, cache: str) -> np.ndarray:
    """``N0`` such that the process's own ``E[n n']`` equals it (a fixed point,
    iterated four times on 150,000-row pilot runs, seed 493001; started at the
    standard normal orthant value ``diag 1/2, offdiag rho/4 + sqrt(1 - rho^2)/(2 pi)``).
    The estimators never see it: they use the sample ``N``. It only keeps the
    truth's intercept consistent. Cached."""
    key = f"k{k}_nu{'inf' if nu is None else int(nu)}"
    path = os.path.join(cache, "adcc_n0.json")
    store = json.load(open(path)) if os.path.exists(path) else {}
    if key in store:
        return np.array(store[key]["n0"])
    rho = ADCC_RHO_BAR
    n0 = np.full((k, k), rho / 4.0 + math.sqrt(1.0 - rho * rho) / (2.0 * math.pi))
    np.fill_diagonal(n0, 0.5)
    hist = []
    for _ in range(4):
        _, z = simulate_adcc(150_000, 493001, k, ADCC_TRUTH, rho, n0, nu)
        new = n_mean(z)
        hist.append(float(np.max(np.abs(new - n0))))
        n0 = new
    store[key] = {"n0": n0.tolist(), "max_change_by_iteration": hist, "zz_mean": (z.T @ z / z.shape[0]).tolist()}
    with open(path, "w") as fh:
        json.dump(store, fh, indent=1)
    return n0


# ---- the batch side: the likelihood written by hand, scipy computes the oracle


def q_path_adcc(a: float, b: float, g: float, z: np.ndarray, qbar: np.ndarray, nmat: np.ndarray) -> np.ndarray:
    """``R_t`` for every row, ``Q_0 = Qbar``, by ``lfilter`` (the input of the
    recursion is data only: ``C + a z z' + g n n'`` of the previous row)."""
    from scipy.signal import lfilter

    n, k = z.shape
    nv = neg_part(z)
    zz = z[:, :, None] * z[:, None, :]
    nn = nv[:, :, None] * nv[:, None, :]
    u = np.empty((n, k, k))
    u[0] = qbar
    u[1:] = (1.0 - a - b) * qbar - g * nmat + a * zz[:-1] + g * nn[:-1]
    q = lfilter([1.0], [1.0, -b], u, axis=0)
    d = 1.0 / np.sqrt(np.einsum("tii->ti", q))
    return q * d[:, :, None] * d[:, None, :]


def fit_adcc(z: np.ndarray, qbar: np.ndarray, nmat: np.ndarray, lik: str) -> dict:
    """The oracle: ``(a, b, g)`` (``lik="normal"``) or ``(a, b, g, nu)``
    (``"t"``) maximizing the second-stage log-likelihood by scipy's SLSQP,
    from ``(0.03, 0.85, 0.02[, 8])``, ``a, b, g >= 0``, ``a + b + delta g <=
    1 - 1e-4``, ``nu`` in [2.1, 200], ``ftol`` 1e-12; sandwich SEs
    ``H^-1 J'J H^-1`` by statsmodels' numerical differentiation (the second
    stage only, as study 485 §14)."""
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_fprime, approx_hess

    delta = delta_of(qbar, nmat)
    p = 3 if lik == "normal" else 4

    def rows(th: np.ndarray) -> np.ndarray:
        rm = q_path_adcc(th[0], th[1], th[2], z, qbar, nmat)
        if lik == "normal":
            return dcc.loglik_rows_k(rm, z)
        return dcc.loglik_rows_t(rm, z, float(th[3]))

    def neg(th: np.ndarray) -> float:
        a, b, g = (float(x) for x in th[:3])
        if a < 0.0 or b < 0.0 or g < 0.0 or a + b + delta * g >= 1.0 or (p == 4 and th[3] <= 2.0):
            return dcc.INFEASIBLE
        v = -float(np.sum(rows(th)))
        return v if math.isfinite(v) else dcc.INFEASIBLE

    x0 = np.array([0.03, 0.85, 0.02] + ([8.0] if p == 4 else []))
    bounds = [(0.0, 1.0)] * 3 + ([(2.1, 200.0)] if p == 4 else [])
    fit = minimize(
        neg,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": lambda t: 1.0 - ADCC_EPS - t[0] - t[1] - delta * t[2]}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    th = fit.x
    hess = approx_hess(th, lambda t: float(np.sum(rows(t))))
    scores = approx_fprime(th, rows)
    hinv = np.linalg.inv(-hess)
    cov = hinv @ (scores.T @ scores) @ hinv
    return {
        "theta": th,
        "se": np.sqrt(np.diag(cov)),
        "success": bool(fit.success),
        "nit": int(fit.nit),
        "llf": -float(fit.fun),
        "delta": delta,
    }


def scipy_likelihood_rows(rmat: np.ndarray, z: np.ndarray, lik: str, nu: float = 0.0) -> np.ndarray:
    """The same per-row log-likelihood computed by ``scipy.stats``, one call
    per row: ``multivariate_normal.logpdf(z; R_t)`` less the independent
    ``logpdf(z; I)`` (the second-stage likelihood subtracts the margins'
    part), or ``multivariate_t.logpdf(z; scale = R_t (nu - 2) / nu, df = nu)``
    (unit-variance t). Used as the check of the vectorized hand-written one."""
    from scipy.stats import multivariate_normal, multivariate_t

    n, k = z.shape
    out = np.empty(n)
    if lik == "normal":
        ind = multivariate_normal.logpdf(z, mean=np.zeros(k), cov=np.eye(k))
        for t in range(n):
            out[t] = multivariate_normal.logpdf(z[t], mean=np.zeros(k), cov=rmat[t]) - ind[t]
    else:
        for t in range(n):
            out[t] = multivariate_t.logpdf(z[t], loc=np.zeros(k), shape=rmat[t] * (nu - 2.0) / nu, df=nu)
    return out


# ---- the batch start's objective (a, b, g, nu), analytic score, for the core's optimizer


def adcc_t_objective(z: np.ndarray, qbar: np.ndarray, nmat: np.ndarray):  # noqa: ANN201
    """``batch_core.dcc_t_objective`` with the asymmetric term: the
    multivariate-t second-stage log-likelihood in ``(a, b, g, nu)``, ``Q_0 =
    Qbar``, the three sensitivities ``dQ/da = zz' - Qbar + b dQ'/da``, ``dQ/db
    = Q' - Qbar + b dQ'/db``, ``dQ/dg = nn' - N + b dQ'/dg``, the score in
    ``(a, b, g)`` ``1/2 tr((w u u' - R^-1) dR)``, ``w = (nu + k)/(nu - 2 + s)``,
    ``dR = D^-1/2 dQ D^-1/2 - 1/2 (L R + R L)``, in ``nu`` ``dcc_t_row``'s;
    its Gauss-Newton fallback the sum of the rows' score outer products."""
    import batch_core

    n, k = z.shape
    nv = neg_part(z)
    zz = z[:, :, None] * z[:, None, :]
    nn = nv[:, :, None] * nv[:, None, :]

    def f(theta: np.ndarray, grad: bool):  # noqa: ANN202
        a, b, g, nu = (float(x) for x in theta)
        gv = np.zeros(4)
        H = np.zeros((4, 4))
        if not nu > 2.0:
            return -math.inf, gv, H
        c0 = (1.0 - a - b) * qbar - g * nmat
        q = qbar.copy()
        dq = [np.zeros((k, k)) for _ in range(3)]
        ll = 0.0
        for t in range(n):
            if t > 0:
                q_prev = q
                q = c0 + a * zz[t - 1] + g * nn[t - 1] + b * q_prev
                if grad:
                    dq = [
                        zz[t - 1] - qbar + b * dq[0],
                        q_prev - qbar + b * dq[1],
                        nn[t - 1] - nmat + b * dq[2],
                    ]
            dg = np.diag(q)
            if not np.all(dg > 0.0):
                return -math.inf, gv, H
            d = 1.0 / np.sqrt(dg)
            rmat = q * np.outer(d, d)
            lr, s, dnu = batch_core.dcc_t_row(z[t], rmat, nu)
            if not math.isfinite(lr):
                return -math.inf, gv, H
            ll += lr
            if grad:
                rinv = np.linalg.inv(rmat)
                u = rinv @ z[t]
                m = (nu + k) / (nu - 2.0 + s) * np.outer(u, u) - rinv
                row = np.empty(4)
                for i, dqi in enumerate(dq):
                    lam = np.diag(dqi) / dg
                    dr = dqi * np.outer(d, d) - 0.5 * (lam[:, None] * rmat + rmat * lam[None, :])
                    row[i] = 0.5 * float(np.sum(m * dr))
                row[3] = dnu
                gv += row
                H += np.outer(row, row)
        return ll, gv, H

    return f


ADCC_T_GRID = [(a, b, g, v) for a, b in ADCC_GRID_A_B for g in (0.01, 0.05) for v in (4.0, 8.0, 20.0)]


@dataclasses_dataclass
class OnlineAdccT(dcc.OnlineDccT):
    """``dcc.OnlineDccT`` with the asymmetric term. State beyond the DCC's:
    ``N`` (the running mean of ``n n'``), ``dQ/dg``, ``g`` and a 3-by-3
    information matrix; ``n_{t-1}`` is derived from the stored ``z_{t-1}``.
    ``update(z)`` takes one row of standardized residuals; the row's
    correlation is emitted before ``z`` is read."""

    nmat: np.ndarray = dataclasses_field(init=False)
    dqg: np.ndarray = dataclasses_field(init=False)
    n_repaired: int = 0
    last_delta: float = math.nan

    def __post_init__(self) -> None:
        super().__post_init__()
        k = self.k
        self.theta = np.array([self.a0, self.b0, 0.02])
        self.nmat = np.zeros((k, k))
        self.dqg = np.zeros((k, k))
        self.H = np.zeros((3, 3))

    def _fit_buffer(self, zb: np.ndarray) -> None:
        import batch_core

        qbar = zb.T @ zb / zb.shape[0]
        nmat = n_mean(zb)
        delta = delta_of(qbar, nmat)

        def proj(th: np.ndarray) -> np.ndarray:
            p3 = project_adcc(th[:3], delta)
            return np.array([p3[0], p3[1], p3[2], min(max(float(th[3]), batch_core.NU_MIN), batch_core.NU_MAX)])

        self.batch_fit = batch_core.maximize(adcc_t_objective(zb, qbar, nmat), proj, ADCC_T_GRID)
        th = self.batch_fit.theta
        self.theta = project_adcc(th[:3], delta)
        self.nu = float(th[3])

    def _score(self, z: np.ndarray, rmat: np.ndarray, d: np.ndarray, dg: np.ndarray):  # noqa: ANN202
        k, nu = self.k, self.nu
        rinv = np.linalg.inv(rmat)
        u = rinv @ z
        s = float(z @ u)
        m = (nu + k) / (nu - 2.0 + s) * np.outer(u, u) - rinv
        drs = []
        for dq in (self.dqa, self.dqb, self.dqg):
            lam = np.diag(dq) / dg
            drs.append(dq * np.outer(d, d) - 0.5 * (lam[:, None] * rmat + rmat * lam[None, :]))
        g = np.array([0.5 * float(np.sum(m * dr)) for dr in drs])
        prods = [rinv @ dr for dr in drs]
        A = np.array([[0.5 * float(np.sum(pi * pj.T)) for pj in prods] for pi in prods])
        tr = np.array([float(np.trace(pi)) for pi in prods])
        info = (nu + k) / (nu + k + 2.0) * A - np.outer(tr, tr) / (2.0 * (nu + k + 2.0))
        return g, info

    def _update(self, z: np.ndarray) -> None:
        a, b, g = self.theta
        z = np.asarray(z, dtype=float)
        if self.n_seen == 0:
            self.q = np.eye(self.k)  # as the engine: no earlier row
        else:
            zz = np.outer(self.prev_z, self.prev_z)
            pn = neg_part(self.prev_z)
            nn = np.outer(pn, pn)
            q_prev = self.q
            q = (1.0 - a - b) * self.qbar - g * self.nmat + a * zz + g * nn + b * q_prev
            if not (np.all(np.diag(q) > 0.0) and np.linalg.slogdet(q)[0] > 0.0):
                # a replayed row's running Qbar and N can leave the intercept
                # indefinite for a few rows; fall back to the symmetric form
                # and count it (reported; zero after the start in every case)
                q = (1.0 - a - b) * self.qbar + a * zz + b * q_prev
                self.n_repaired += 1
            self.q = q
            self.dqa = zz - self.qbar + b * self.dqa
            self.dqb = q_prev - self.qbar + b * self.dqb
            self.dqg = nn - self.nmat + b * self.dqg
        dg = np.diag(self.q)
        d = 1.0 / np.sqrt(dg)
        rmat = self.q * np.outer(d, d)
        self.pred.append(rmat[np.triu_indices(self.k, 1)])
        gs, info = self._score(z, rmat, d, dg)
        self.last_g, self.last_info = gs, info
        self.n_seen += 1
        if self.n_seen <= self.warm:
            self.H += (info - self.H) / self.n_seen
        else:
            self.n_steps += 1
            gamma = 1.0 / (self.n_steps + self.t0)
            self.H += gamma * (info - self.H)
            proposal = self.theta + gamma * np.linalg.solve(self.H, gs)
            self.last_delta = delta_of(self.qbar, self.nmat)
            projected = project_adcc(proposal, self.last_delta)
            self.n_projected += int(not np.allclose(projected, proposal, rtol=0.0, atol=1e-15))
            self.theta = projected
        self.qbar += (np.outer(z, z) - self.qbar) / self.n_seen
        nz = neg_part(z)
        self.nmat += (np.outer(nz, nz) - self.nmat) / self.n_seen
        self.prev_z = z.copy()


def adcc_distinct_floats(k: int) -> int:
    """The estimator's state beyond the margins, in floats (the distinct
    entries of each symmetric matrix): ``Qbar``, ``N``, ``Q`` and the three
    sensitivities (6 matrices of ``k (k + 1) / 2``), the last ``z`` (``k``),
    ``theta`` (3), the information (6), ``nu`` and the row count: ``3 k (k + 1)
    + k + 11``."""
    return 6 * (k * (k + 1) // 2) + k + 3 + 6 + 1 + 1


MARGIN_STATE_BYTES = 128  # one targeted online GARCH(1,1) (study 485, dcc.py's docstring)
ADCC_KS = (2, 3, 5, 10, 11, 20, 21, 50, 70, 71, 100, 101, 140)


def adcc_bytes(k: int) -> dict:
    """The group's state in bytes against ``k``: the correlation stage
    (``8 (3 k (k + 1) + k + 11) = 24 k^2 + 32 k + 88``), the ``k`` margins
    (128 B each), and the shared start's buffers, held until the start is fitted:
    each margin's 500 rows of ``r``, its 500 in-sample variances (kept in this
    implementation, not freed), and the stage's 500 rows of ``z``: ``12,000 k``
    bytes. ``steady`` is the state after the start; ``peak`` includes the buffers."""
    stage = 8 * adcc_distinct_floats(k)
    steady = stage + MARGIN_STATE_BYTES * k
    buffers = 8 * ADCC_START * 3 * k
    return {"k": k, "stage": stage, "margins": MARGIN_STATE_BYTES * k, "steady": steady,
            "buffers": buffers, "peak": steady + buffers}


def adcc_caps() -> dict:
    rows = []
    for k in ADCC_KS:
        b = adcc_bytes(k)
        b["steady_verdict"] = cap_verdict(b["steady"])
        b["peak_verdict"] = cap_verdict(b["peak"])
        rows.append(b)
    def first_k(key: str, limit: int) -> int:
        k = 1
        while adcc_bytes(k)[key] < limit:
            k += 1
        return k
    return {
        "rows": rows,
        "limits": {
            "steady_warn_k": first_k("steady", WARN_BYTES),
            "steady_refuse_k": first_k("steady", CAP_BYTES),
            "peak_warn_k": first_k("peak", WARN_BYTES),
            "peak_refuse_k": first_k("peak", CAP_BYTES),
        },
    }


def adcc_case(k: int, noise: str, seed: int, n: int = ADCC_ROWS) -> dict:
    """One seed: simulate, the oracles (Gaussian and t), the online engine."""
    nu_true = ADCC_NU if noise == "t5" else None
    n0 = calibrate_n0(k, nu_true, OUT)
    t_all = time.perf_counter()
    r, z_true = simulate_adcc(n, seed, k, ADCC_TRUTH, ADCC_RHO_BAR, n0, nu_true)
    out: dict = {"spec": f"adcc_k{k}_{noise}", "seed": seed, "k": k, "noise": noise, "n": n}
    # --- the oracles: arch's margins, z from the in-sample volatility
    t0 = time.perf_counter()
    z_b = dcc.batch_margin_z(r)
    qbar_b = z_b.T @ z_b / n
    n_b = n_mean(z_b)
    out["delta"] = delta_of(qbar_b, n_b)
    fits = {lik: fit_adcc(z_b, qbar_b, n_b, lik) for lik in ("normal", "t")}
    out["t_oracle_s"] = time.perf_counter() - t0
    iu = np.triu_indices(k, 1)
    # --- the engine: shared start, t second stage
    t0 = time.perf_counter()
    rem = dcc.REMEDIES["shared500core_t"]
    mkw = {"target": "running", **rem["margins"]}
    margin_est = [garch.run(r[:, i], **mkw) for i in range(k)]
    s2 = np.column_stack([np.asarray(m.pred) for m in margin_est])
    rows = np.flatnonzero(np.all(np.isfinite(s2), axis=1))
    est = OnlineAdccT(k=k, **rem["stage2"])
    zbuf = np.column_stack(
        [r[: len(m.replay_sigma2), i] / np.sqrt(np.asarray(m.replay_sigma2)) for i, m in enumerate(margin_est)]
    )
    est.shared_start(zbuf)
    for row in r[rows] / np.sqrt(s2[rows]):
        est.update(row)
    out["t_online_s"] = time.perf_counter() - t0
    th_o = est.theta
    pred = np.full((n, len(iu[0])), np.nan)
    pred[rows] = np.asarray(est.pred)
    tail = slice(n // 2, n)
    out.update(
        silent_rows=int(n - rows.size),
        first_dcc_row=int(rows[0]),
        online_steps_projected=int(est.n_projected),
        online_q_repaired=int(est.n_repaired),
        online_delta_last=float(est.last_delta),
        online_in_region=bool(th_o[0] + th_o[1] + est.last_delta * th_o[2] < 1.0),
        nu_online_start=float(est.nu),
        start_fit_stop=est.batch_fit.stop,
        start_fit_iterations=est.batch_fit.iterations,
        start_fit_evaluations=est.batch_fit.evaluations,
        start_fit_theta=[float(x) for x in est.batch_fit.theta],
        state_floats_distinct=adcc_distinct_floats(k),
    )
    for i, nm in enumerate(ADCC_NAMES):
        out[f"{nm}_online"] = float(th_o[i])
    for lik, tag in (("normal", "g"), ("t", "t")):
        f = fits[lik]
        th, se = f["theta"], f["se"]
        rm_b = q_path_adcc(th[0], th[1], th[2], z_b, qbar_b, n_b)
        rho_b = rm_b[:, iu[0], iu[1]]
        gap = np.median(np.abs(pred[tail] - rho_b[tail]), axis=0)
        # the online parameters on the oracle's z, in the oracle's likelihood
        rm_o = q_path_adcc(th_o[0], th_o[1], th_o[2], z_b, qbar_b, n_b)
        if lik == "normal":
            ll_o = float(np.sum(dcc.loglik_rows_k(rm_o, z_b)))
            hand = dcc.loglik_rows_k(rm_b, z_b)
        else:
            ll_o = float(np.sum(dcc.loglik_rows_t(rm_o, z_b, float(th[3]))))
            hand = dcc.loglik_rows_t(rm_b, z_b, float(th[3]))
        t1 = time.perf_counter()
        sp = scipy_likelihood_rows(rm_b, z_b, lik, float(th[3]) if lik == "t" else 0.0)
        out[f"{tag}_scipy_check_s"] = time.perf_counter() - t1
        out[f"{tag}_scipy_row_maxabs"] = float(np.max(np.abs(sp - hand)))
        out[f"{tag}_scipy_total_diff"] = float(abs(np.sum(sp) - np.sum(hand)))
        out[f"{tag}_batch_success"] = f["success"]
        out[f"{tag}_batch_nit"] = f["nit"]
        out[f"{tag}_batch_at_constraint"] = bool(th[0] + th[1] + f["delta"] * th[2] >= 1.0 - ADCC_EPS - 1e-6 or min(th[:3]) <= 1e-6)
        out[f"{tag}_llf_batch"] = f["llf"]
        out[f"{tag}_llf_diff_per_row"] = (f["llf"] - ll_o) / n
        out[f"{tag}_rho_gap_worst_pair_median"] = float(gap.max())
        for i, nm in enumerate(ADCC_NAMES):
            out[f"{tag}_{nm}_batch"] = float(th[i])
            out[f"{tag}_{nm}_se"] = float(se[i])
            out[f"{tag}_{nm}_gap_se"] = float((th_o[i] - th[i]) / se[i])
            out[f"{tag}_{nm}_gap_abs"] = float(th_o[i] - th[i])
        if lik == "t":
            out["t_nu_batch"] = float(th[3])
            out["t_nu_se"] = float(se[3])
    out["t_case_s"] = time.perf_counter() - t_all
    return out


def _adcc_case_args(a):  # noqa: ANN001, ANN202
    return adcc_case(*a)


def run_adcc_cases(path: str, cases: list[tuple], workers: int) -> list[dict]:
    done = {(r["spec"], r["seed"]) for r in base._load(path)}
    todo = [c for c in cases if (f"adcc_k{c[0]}_{c[1]}", c[2]) not in done]
    if todo:
        print(f"{path}: {len(done)} cached, {len(todo)} to run", flush=True)
        # the calibration is cached by the first call, before the workers fork
        for k, noise in sorted({(c[0], c[1]) for c in todo}):
            calibrate_n0(k, ADCC_NU if noise == "t5" else None, OUT)
        from multiprocessing import get_context

        with get_context("fork").Pool(workers) as pool, open(path, "a") as fh:
            for i, res in enumerate(pool.imap_unordered(_adcc_case_args, todo)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
    return base._load(path)


def adcc_count(rows: list[dict], tag: str) -> dict:
    """The criterion for one oracle (``tag`` ``g``: Gaussian, ``t``: Student-t):
    seeds with ``a``, ``b`` and ``g`` all within 2 of the oracle's SEs, and
    study 485 section 24.3's per-seed bounds (every parameter within 4 SE;
    worst-pair correlation gap at most 0.03)."""
    n = len(rows)
    gaps = np.array([[abs(r[f"{tag}_{nm}_gap_se"]) for nm in ADCC_NAMES] for r in rows]) if n else np.zeros((0, 3))
    within2 = int(np.sum(np.all(gaps <= 2.0, axis=1))) if n else 0
    within4 = int(np.sum(np.all(gaps <= 4.0, axis=1))) if n else 0
    rho = np.array([r[f"{tag}_rho_gap_worst_pair_median"] for r in rows]) if n else np.zeros(0)
    wi = int(np.argmax(gaps.max(axis=1))) if n else 0
    return {
        "cases": n,
        "within2": within2,
        "within4": within4,
        "rho_gap_le_003": int(np.sum(rho <= 0.03)) if n else 0,
        "rho_gap_max": float(rho.max()) if n else math.nan,
        "median_gap_se": {nm: float(np.median(gaps[:, i])) for i, nm in enumerate(ADCC_NAMES)} if n else {},
        "max_gap_se": {nm: float(gaps[:, i].max()) for i, nm in enumerate(ADCC_NAMES)} if n else {},
        "worst_seed": int(rows[wi]["seed"]) if n else None,
        "worst_gap_se": float(gaps.max()) if n else math.nan,
        "max_gap_abs": {nm: float(max(abs(r[f"{tag}_{nm}_gap_abs"]) for r in rows)) for nm in ADCC_NAMES} if n else {},
        "median_se": {nm: float(np.median([r[f"{tag}_{nm}_se"] for r in rows])) for nm in ADCC_NAMES} if n else {},
        "llf_gap_per_row_median": float(np.median([r[f"{tag}_llf_diff_per_row"] for r in rows])) if n else math.nan,
        "llf_gap_per_row_max": float(max(r[f"{tag}_llf_diff_per_row"] for r in rows)) if n else math.nan,
        "batch_success": int(sum(r[f"{tag}_batch_success"] for r in rows)),
        "batch_at_constraint": int(sum(r[f"{tag}_batch_at_constraint"] for r in rows)),
    }


def adcc_checks() -> dict:
    """Checks, on one series per setting: the reduction (``g = 0`` gives study
    485's DCC path), the asymmetric term's definition, the analytic gradient of
    the start's objective and the online score against central differences, the
    out-of-sample property, and the hand-written likelihood against scipy.stats
    (also reported per case)."""
    out: dict = {}
    k = 2
    n0 = calibrate_n0(k, None, OUT)
    r, z = simulate_adcc(3000, 493, k, ADCC_TRUTH, ADCC_RHO_BAR, n0, None)
    qbar = z.T @ z / z.shape[0]
    nmat = n_mean(z)
    out["n0"] = n0.tolist()
    out["n0_orthant_closed_form_offdiag"] = ADCC_RHO_BAR / 4.0 + math.sqrt(1.0 - ADCC_RHO_BAR**2) / (2.0 * math.pi)
    out["delta_of_truth_process"] = delta_of(np.full((2, 2), 0.5) + 0.5 * np.eye(2), n0)
    # (1) g = 0 reduces to study 485's DCC path
    rm = q_path_adcc(0.05, 0.9, 0.0, z, qbar, nmat)
    rd = dcc.r_path_k(0.05, 0.9, z, qbar)
    out["g0_equals_dcc_max_abs"] = float(np.max(np.abs(rm - rd)))
    # (2) the definition, row by row, against a direct loop
    a, b, g = 0.04, 0.9, 0.03
    q = qbar.copy()
    worst = 0.0
    rmat = q_path_adcc(a, b, g, z, qbar, nmat)
    for t in range(1, 200):
        zp = z[t - 1]
        npv = zp * (zp < 0)
        q = qbar - a * qbar - b * qbar - g * nmat + a * np.outer(zp, zp) + b * q + g * np.outer(npv, npv)
        d = 1.0 / np.sqrt(np.diag(q))
        worst = max(worst, float(np.max(np.abs(q * np.outer(d, d) - rmat[t]))))
    out["definition_loop_vs_lfilter_max_abs"] = worst
    # (3) the start objective's gradient against central differences of its value
    zb = z[:300]
    qb, nb = zb.T @ zb / 300, n_mean(zb)
    obj = adcc_t_objective(zb, qb, nb)
    th = np.array([0.04, 0.9, 0.03, 7.0])
    _, g_an, _ = obj(th, True)
    num = np.zeros(4)
    for j in range(4):
        h = 1e-6
        tp, tm = th.copy(), th.copy()
        tp[j] += h
        tm[j] -= h
        num[j] = (obj(tp, False)[0] - obj(tm, False)[0]) / (2 * h)
    out["start_objective_gradient_rel_err"] = float(np.max(np.abs(g_an - num)) / np.max(np.abs(num)))
    # (4) the online row score against central differences of the row log-likelihood
    est = OnlineAdccT(k=k, warm=10**9, t0=10.0)
    est.theta = np.array([0.04, 0.9, 0.03])
    est.nu = 7.0
    est.started = True
    for x in z[:200]:
        est._update(x)
    # the t row log density at the engine's own Q path as a function of theta, replayed
    def row_ll(theta: np.ndarray) -> float:
        e = OnlineAdccT(k=k, warm=10**9, t0=10.0)
        e.theta = np.array(theta, dtype=float)
        e.nu = 7.0
        e.started = True
        for x in z[:200]:
            e._update(x)
        # the last row's rho and density
        rm_last = np.eye(k)
        rm_last[0, 1] = rm_last[1, 0] = e.pred[-1][0]
        s = float(z[199] @ np.linalg.solve(rm_last, z[199]))
        return float(
            math.lgamma(0.5 * (7.0 + k)) - math.lgamma(3.5) - 0.5 * k * math.log(math.pi * 5.0)
            - 0.5 * np.linalg.slogdet(rm_last)[1] - 0.5 * (7.0 + k) * math.log1p(s / 5.0)
        )
    # the engine's last-row score is for z[199] at the Q path that used z[:199]; compare its g
    gnum = np.zeros(3)
    base_th = np.array([0.04, 0.9, 0.03])
    for j in range(3):
        h = 1e-6
        tp, tm = base_th.copy(), base_th.copy()
        tp[j] += h
        tm[j] -= h
        gnum[j] = (row_ll(tp) - row_ll(tm)) / (2 * h)
    out["online_score_rel_err"] = float(np.max(np.abs(est.last_g - gnum)) / np.max(np.abs(gnum)))
    # (5) out of sample: change row 150, earlier emitted correlations are unchanged
    e1, e2 = OnlineAdccT(k=k, warm=50, t0=50.0), OnlineAdccT(k=k, warm=50, t0=50.0)
    for t in range(300):
        e1.update(z[t])
        e2.update(z[t] + (5.0 if t == 150 else 0.0))
    out["out_of_sample_max_move_up_to_changed_row"] = float(
        max(np.max(np.abs(e1.pred[t] - e2.pred[t])) for t in range(151))
    )
    # (6) the hand-written likelihood against scipy.stats
    rm = q_path_adcc(0.04, 0.9, 0.03, z, qbar, nmat)
    for lik in ("normal", "t"):
        hand = dcc.loglik_rows_k(rm, z) if lik == "normal" else dcc.loglik_rows_t(rm, z, 5.0)
        sp = scipy_likelihood_rows(rm[:1000], z[:1000], lik, 5.0)
        out[f"scipy_{lik}_row_maxabs_1000_rows"] = float(np.max(np.abs(sp - hand[:1000])))
    out["allocated_distinct_state_floats_k2_k3"] = [adcc_distinct_floats(2), adcc_distinct_floats(3)]
    return out


def adcc_part(k: int, args) -> dict:  # noqa: ANN001
    seeds = SEEDS[: args.seeds]
    path = os.path.join(args.out, f"adcc_k{k}.jsonl")
    cases = [(k, noise, s) for noise in ("normal", "t5") for s in seeds]
    rows = base._load(path) if args.no_run else run_adcc_cases(path, cases, args.workers)
    res: dict = {"k": k, "seeds": [seeds[0], seeds[-1]], "n": ADCC_ROWS, "truth": list(ADCC_TRUTH)}
    if k == 2:
        res["checks"] = adcc_checks()
    res["state_caps"] = adcc_caps()
    for noise in ("normal", "t5"):
        sub = sorted((r for r in rows if r["noise"] == noise), key=lambda r: r["seed"])
        d: dict = {"vs_gaussian_oracle": adcc_count(sub, "g"), "vs_t_oracle": adcc_count(sub, "t")}
        if sub:
            d["scipy_row_maxabs"] = {
                tg: float(max(r[f"{tg}_scipy_row_maxabs"] for r in sub)) for tg in ("g", "t")
            }
            d["scipy_total_diff_max"] = {
                tg: float(max(r[f"{tg}_scipy_total_diff"] for r in sub)) for tg in ("g", "t")
            }
            d["silent_rows"] = sorted({r["silent_rows"] for r in sub})
            d["online_steps_projected_seeds"] = int(sum(r["online_steps_projected"] > 0 for r in sub))
            d["online_steps_projected_max"] = int(max(r["online_steps_projected"] for r in sub))
            d["online_q_repaired_max"] = int(max(r["online_q_repaired"] for r in sub))
            d["online_in_region"] = int(sum(r["online_in_region"] for r in sub))
            d["start_fit_stops"] = {
                s_: int(sum(r["start_fit_stop"] == s_ for r in sub)) for s_ in ("converged", "no ascent", "iteration cap")
            }
            d["nu_online_start_median"] = float(np.median([r["nu_online_start"] for r in sub]))
            d["nu_online_start_range"] = [
                float(min(r["nu_online_start"] for r in sub)),
                float(max(r["nu_online_start"] for r in sub)),
            ]
            d["nu_oracle_median"] = float(np.median([r["t_nu_batch"] for r in sub]))
            d["delta_median"] = float(np.median([r["delta"] for r in sub]))
            d["t_online_median_s"] = float(np.median([r["t_online_s"] for r in sub]))
            d["t_case_median_s"] = float(np.median([r["t_case_s"] for r in sub]))
            d["batch_param_median"] = {
                nm: float(np.median([r[f"g_{nm}_batch"] for r in sub])) for nm in ADCC_NAMES
            }
            d["online_param_median"] = {
                nm: float(np.median([r[f"{nm}_online"] for r in sub])) for nm in ADCC_NAMES
            }
        res[noise] = d
    have = {(r["spec"], r["seed"]) for r in rows}
    res["complete"] = all((f"adcc_k{c[0]}_{c[1]}", c[2]) in have for c in cases)
    return res


def summarise_adcc(res: dict) -> None:
    if "checks" in res:
        for key, v in res["checks"].items():
            print("check", key, v)
    print("complete:", res["complete"], "k =", res["k"], "truth", res["truth"])
    sc = res.get("state_caps")
    if sc:
        print("state caps:", sc["limits"])
        for r in sc["rows"]:
            print("  ", r)
    for noise in ("normal", "t5"):
        d = res[noise]
        for orc in ("vs_gaussian_oracle", "vs_t_oracle"):
            c = d[orc]
            print(
                f"{noise:6s} {orc:20s}: within2 {c['within2']}/{c['cases']}, within4 {c['within4']}, "
                f"rho gap<=0.03 {c['rho_gap_le_003']} (max {c['rho_gap_max']:.4f}); worst {c['worst_gap_se']:.3g} SE "
                f"(seed {c['worst_seed']}); max gap in SE {c['max_gap_se']}; max abs {c['max_gap_abs']}; "
                f"median SE {c['median_se']}; llf gap/row median {c['llf_gap_per_row_median']:.2g} max "
                f"{c['llf_gap_per_row_max']:.2g}; SLSQP ok {c['batch_success']}; at constraint {c['batch_at_constraint']}"
            )
        for key in ("scipy_row_maxabs", "scipy_total_diff_max", "silent_rows", "online_steps_projected_seeds",
                    "online_steps_projected_max", "online_q_repaired_max", "online_in_region", "start_fit_stops",
                    "nu_online_start_median", "nu_online_start_range", "nu_oracle_median", "delta_median",
                    "t_online_median_s", "t_case_median_s", "batch_param_median", "online_param_median"):
            print("   ", key, d.get(key))


PARTS = {
    "part1": (lambda a: ets_part("part1", a), summarise_ets),
    "part2": (lambda a: ets_part("part2", a), summarise_ets),
    "part3": (lambda a: adcc_part(2, a), summarise_adcc),
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
