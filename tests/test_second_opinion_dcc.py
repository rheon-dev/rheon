"""`dcc` against the batch quasi-likelihood fit (task 493): the online
estimator, through the bank, against the maximum of Engle's (2002)
second-stage likelihood over all the same rows.

No mainstream Python library fits DCC (``arch`` has no multivariate
model; ``mgarch`` on PyPI does not compute the likelihood it names), so the
oracle is the definition written by hand and a general optimizer:

- **the definition**: ``Q_1 = Q̄``, ``Q_{t+1} = (1 - a - b) Q̄ - g N̄ + a z_t
  z_tᵀ + g n_t n_tᵀ + b Q_t``, ``Q̄`` and ``N̄`` the sample means of ``z zᵀ``
  and ``n nᵀ`` over every row, each entry of ``Q`` run by
  ``scipy.signal.lfilter``; ``R_t`` read off ``Q_t``; the row's
  quasi-log-likelihood ``-½ (ln |R_t| + z_tᵀ R_t⁻¹ z_t)``;
- **the optimizer**: ``scipy.optimize.minimize(method="SLSQP")`` over ``a,
  b, g ≥ 0`` and ``a + b + δ g ≤ 1 - 1e-4``, ``δ`` the largest generalized
  eigenvalue of ``(N̄, Q̄)`` by ``scipy.linalg.eigh``, from a grid's best
  point; the likelihood ``1e12`` below where ``Q`` leaves the positive
  definite cone;
- **scipy's density** certifies the hand-written likelihood at the optimum:
  ``scipy.stats.multivariate_normal.logpdf`` under each ``R_t``, less the
  constant, row by row;
- **standard errors**: the second stage's sandwich ``H⁻¹ JᵀJ H⁻¹`` by
  statsmodels' numerical derivatives of the rows' likelihoods at the
  optimum. They leave out the margins' estimation error (none here: the
  rows are standardized residuals, ``margins="none"``).

The two differ by construction in the targets: the batch fit's are the
sample means of every row, the online one's the running means of the rows
before. The comparisons, each a seed's: each parameter's gap in the
oracle's standard errors; the worst pair's median ``|ρ_online - ρ_batch|``
over the second half of the rows, the batch path at the batch parameters;
and ``|Δ(a + b)|``. The bounds are from a 50-seed grid of the build, each
stated beside its test.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from arch import arch_model
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import multivariate_normal
from statsmodels.tools.numdiff import approx_fprime, approx_hess

import ledger
from rheon import polars as rp

TIER = "mixed"

CAP = 1.0 - 1e-4
NAMES = ["a", "b", "c"]
#: (series, truth (a, b, g), Student-t degrees of freedom or None)
FORMS = {
    "dcc2": (2, (0.05, 0.90, 0.0), None),
    "dcc3": (3, (0.05, 0.93, 0.0), None),
    "adcc2": (2, (0.03, 0.93, 0.03), None),
    "adcc3": (3, (0.03, 0.93, 0.03), None),
    "adcc2_t5": (2, (0.03, 0.93, 0.03), 5.0),
    "dcc3_t5": (3, (0.05, 0.93, 0.0), 5.0),
}


def simulate(form: str, n: int, seed: int) -> np.ndarray:
    """``n`` rows of standardized residuals of the form's process, 1,000
    discarded: every target correlation 0.5; the asymmetric term's ``N`` at
    half the target, which keeps the truth's intercept positive definite;
    innovations normal, or a multivariate Student-t at unit variance."""
    k, (a, b, g), nu = FORMS[form]
    rng = np.random.default_rng(493_000 + seed)
    qbar = np.full((k, k), 0.5)
    np.fill_diagonal(qbar, 1.0)
    nbar = 0.5 * qbar
    q = qbar.copy()
    out = np.empty((n, k))
    for t in range(n + 1000):
        d = np.sqrt(np.diag(q))
        x = rng.standard_normal(k)
        if nu is not None:
            x *= math.sqrt((nu - 2.0) / rng.chisquare(nu))
        z = np.linalg.cholesky(q / np.outer(d, d)) @ x
        m = np.minimum(z, 0.0)
        q = (1 - a - b) * qbar - g * nbar + a * np.outer(z, z) + g * np.outer(m, m) + b * q
        if t >= 1000:
            out[t - 1000] = z
    return out


def targets(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neg = np.minimum(z, 0.0)
    return z.T @ z / len(z), neg.T @ neg / len(z)


def q_path(theta, z: np.ndarray, qbar: np.ndarray, nbar: np.ndarray) -> np.ndarray:
    """``Q_t`` for every row: ``Q_1 = Q̄`` and the recursion, each entry by
    ``lfilter``."""
    a, b = theta[0], theta[1]
    g = theta[2] if len(theta) > 2 else 0.0
    neg = np.minimum(z, 0.0)
    x = (
        (1 - a - b) * qbar
        - g * nbar
        + a * np.einsum("ti,tj->tij", z[:-1], z[:-1])
        + g * np.einsum("ti,tj->tij", neg[:-1], neg[:-1])
    )
    rest = lfilter([1.0], [1.0, -b], x, axis=0, zi=(b * qbar)[None])[0]
    return np.concatenate([qbar[None], rest])


def r_path(theta, z, qbar, nbar) -> np.ndarray:
    q = q_path(theta, z, qbar, nbar)
    d = np.sqrt(np.einsum("tii->ti", q))
    return q / (d[:, :, None] * d[:, None, :])


def row_lik(theta, z, qbar, nbar) -> np.ndarray:
    """Each row's quasi-log-likelihood, ``-½ (ln |R| + zᵀR⁻¹z)``; ``-1e12``
    on every row where ``R`` is not positive definite."""
    r = r_path(theta, z, qbar, nbar)
    sign, logdet = np.linalg.slogdet(r)
    if not np.all(sign > 0):
        return np.full(len(z), -1e12)
    quad = np.einsum("ti,ti->t", z, np.linalg.solve(r, z[:, :, None])[:, :, 0])
    return -0.5 * (logdet + quad)


def batch_fit(z: np.ndarray, asymmetric: bool) -> dict:
    qbar, nbar = targets(z)
    delta = float(eigh(nbar, qbar, eigvals_only=True)[-1]) if asymmetric else 0.0
    c = np.array([1.0, 1.0, delta][: 3 if asymmetric else 2])

    def nll(th):
        if np.any(th < 0) or c @ th > CAP:
            return 1e12
        return -float(row_lik(th, z, qbar, nbar).sum())

    grid = [
        np.array([a, b, g][: len(c)])
        for a in (0.02, 0.05)
        for b in (0.85, 0.93)
        for g in (0.01, 0.05)
    ]
    start = min(grid, key=nll)
    res = minimize(
        nll,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(c),
        constraints=[{"type": "ineq", "fun": lambda th: CAP - c @ th}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert res.success, res.message
    th = res.x
    # scipy's density, row by row, certifies the hand-written likelihood
    r = r_path(th, z, qbar, nbar)
    k = z.shape[1]
    rows = np.linspace(0, len(z) - 1, 200).astype(int)
    lik = row_lik(th, z, qbar, nbar)
    for t in rows:
        dens = multivariate_normal(np.zeros(k), r[t]).logpdf(z[t]) + 0.5 * k * math.log(2 * math.pi)
        assert abs(dens - lik[t]) <= 1e-10, (t, dens, lik[t])
    # the second stage's sandwich
    h = approx_hess(th, lambda x: float(row_lik(x, z, qbar, nbar).sum()))
    jac = approx_fprime(th, lambda x: row_lik(x, z, qbar, nbar), centered=True)
    hinv = np.linalg.inv(h)
    cov = hinv @ (jac.T @ jac) @ hinv
    return {"theta": th, "se": np.sqrt(np.diag(cov)), "r": r}


def online(z: np.ndarray, asymmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    k = z.shape[1]
    names = NAMES[:k]
    df = pl.DataFrame({n: z[:, j] for j, n in enumerate(names)})
    spec = rp.spec.dcc(
        "m", targets=names, margins="none", asymmetric=asymmetric, half_life=float("inf")
    )
    bank = rp.ModelBank([spec])
    out = bank.fit_predict(df).unnest("m")
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    corr = np.column_stack([out[f"corr_{names[i]}_{names[j]}"].to_numpy() for i, j in pairs])
    return np.array(bank.coef()["coef"].to_list()), corr


def compare(form: str, seed: int, n: int = 10_000) -> dict:
    k, truth, _ = FORMS[form]
    asymmetric = truth[2] > 0
    z = simulate(form, n, seed)
    b = batch_fit(z, asymmetric)
    th, corr = online(z, asymmetric)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    batch_corr = np.column_stack([b["r"][:, i, j] for i, j in pairs])
    half = slice(n // 2, n)
    rho_gap = max(
        float(np.median(np.abs(corr[half, p] - batch_corr[half, p]))) for p in range(len(pairs))
    )
    gap = (th - b["theta"]) / b["se"]
    return {
        "gap": gap,
        "within2": bool(np.all(np.abs(gap) <= 2.0)),
        "rho_gap": rho_gap,
        "persistence_gap": abs((th[0] + th[1]) - (b["theta"][0] + b["theta"][1])),
        "online": th,
        "batch": b["theta"],
        "se": b["se"],
    }


# --- the essentials: two seeds a form ---------------------------------------


@pytest.mark.parametrize("form", ["dcc2", "adcc3", "adcc2_t5"])
def test_two_seeds_a_form_are_the_batch_fit_to_first_order(form):
    """Each of two seeds at 10,000 rows within the 50-seed grid's per-seed
    bounds below: every parameter's gap in the oracle's standard errors, the
    worst pair's median correlation gap, and the persistence gap."""
    _, worst, rho, pers = BOUNDS[form]
    for seed in (0, 1):
        c = compare(form, seed)
        assert np.max(np.abs(c["gap"])) <= worst, (form, seed, c)
        assert c["rho_gap"] <= rho, (form, seed, c["rho_gap"])
        assert c["persistence_gap"] <= pers, (form, seed, c["persistence_gap"])


# --- the extended grids: 50 seeds a form ------------------------------------

SEEDS = range(50)
#: Per form, the count within 2 SE on every parameter the grid must reach,
#: and the per-seed bounds: the largest gap in SE, the worst pair's median
#: correlation gap and the persistence gap, each the measurement's with
#: headroom.
#: Measured (2026-11-22, 50 seeds a form, the start held to its
#: likelihood-ratio test and the projected step's gain): within 2 SE 50 of
#: 50 on every form; the largest gap 0.73, 0.74, 1.17, 0.91 SE under normal
#: innovations and 1.95, 0.82 under t(5); the correlation gap at most 0.0097
#: and 0.0169; the persistence gap at most 0.0078 and 0.0206. The floors
#: are four seeds under, the per-seed bounds unchanged from the first
#: build's grid (worst 0.99, 1.88), about 1.3 to 1.5 times the worst.
NORMAL = (46, 1.5, 0.02, 0.015)
HEAVY = (46, 3.0, 0.03, 0.03)
BOUNDS: dict[str, tuple[int, float, float, float]] = {
    "dcc2": NORMAL,
    "dcc3": NORMAL,
    "adcc2": NORMAL,
    "adcc3": NORMAL,
    "adcc2_t5": HEAVY,
    "dcc3_t5": HEAVY,
}


#: The hash guard's inputs: two seeds a form, the online fit alone, about 2
#: s. All 300 of the grid's would be 42 s of the gate (each simulation a
#: Python loop of 11,000 rows, 0.1 s, measured 2026-11-23), and a change to
#: the estimator moves these seeds' coefficients as well.
ledger.guard(
    "dcc_batch",
    lambda: [
        online(simulate(f, 10_000, s), FORMS[f][1][2] > 0)[0] for f in sorted(FORMS) for s in (0, 1)
    ],
    oracles=["scipy", "statsmodels"],
)


@pytest.mark.oracle_grid(reason="50 seeds a form at 10,000 rows, a scipy fit and its sandwich each")
@pytest.mark.parametrize("form", sorted(FORMS))
def test_fifty_seeds_a_form_against_the_batch_fit(form):
    floor, worst, rho, pers = BOUNDS[form]
    cs = [compare(form, s) for s in SEEDS]
    within = sum(c["within2"] for c in cs)
    ledger.record("dcc_batch", f"{form}_gap_se", [float(np.max(np.abs(c["gap"]))) for c in cs])
    ledger.record("dcc_batch", f"{form}_rho_gap", [c["rho_gap"] for c in cs])
    ledger.record("dcc_batch", f"{form}_persistence_gap", [c["persistence_gap"] for c in cs])
    ledger.record("dcc_batch", f"{form}_outside_2se", len(cs) - within, exact=True)
    assert within >= floor, (form, within)
    assert max(np.max(np.abs(c["gap"])) for c in cs) <= worst, form
    assert max(c["rho_gap"] for c in cs) <= rho, form
    assert max(c["persistence_gap"] for c in cs) <= pers, form


# --- through GARCH margins ---------------------------------------------------


def returns(z: np.ndarray, seed: int) -> np.ndarray:
    """Each series of ``z`` scaled by its own GARCH(1,1) at ``(0.02, 0.08,
    0.9)``, the study's margins."""
    del seed
    s = np.full(z.shape[1], 1.0)
    e = np.zeros(z.shape[1])
    out = np.empty_like(z)
    for t in range(len(z)):
        s = 0.02 + 0.08 * e * e + 0.9 * s
        e = np.sqrt(s) * z[t]
        out[t] = e
    return out


def compare_with_margins(form: str, seed: int, n: int = 10_000) -> dict:
    """The model on returns, its margins its own GARCH(1,1)s, against the
    two-step batch fit: ``arch``'s GARCH(1,1) of each series, ``z`` from its
    in-sample volatility, and the second stage above on those ``z``."""
    k, truth, _ = FORMS[form]
    asymmetric = truth[2] > 0
    x = returns(simulate(form, n, seed), seed)
    z = np.column_stack(
        [
            x[:, j]
            / arch_model(x[:, j], mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
            .fit(disp="off")
            .conditional_volatility
            for j in range(k)
        ]
    )
    b = batch_fit(z, asymmetric)
    names = NAMES[:k]
    spec = rp.spec.dcc("m", targets=names, asymmetric=asymmetric, half_life=float("inf"))
    bank = rp.ModelBank([spec])
    out = bank.fit_predict(pl.DataFrame({nm: x[:, j] for j, nm in enumerate(names)})).unnest("m")
    th = np.array(bank.coef()["coef"].to_list())
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    half = slice(n // 2, n)
    rho_gap = max(
        float(
            np.median(
                np.abs(out[f"corr_{names[i]}_{names[j]}"].to_numpy()[half] - b["r"][half, i, j])
            )
        )
        for i, j in pairs
    )
    return {"gap": (th - b["theta"]) / b["se"], "rho_gap": rho_gap, "online": th}


@pytest.mark.extended(reason="ten seeds a form at 10,000 rows, arch's margins and a scipy fit each")
@pytest.mark.parametrize("form", ["dcc3", "adcc2"])
def test_through_garch_margins_against_the_two_step_batch_fit(form):
    """The margins add their own differences to the second stage's: the
    online margins' out-of-sample variances against ``arch``'s in-sample
    ones, and their 2,000-row start, which the correlation's own start
    follows. Measured on ten seeds a form (2026-11-22): the largest gap
    1.20 and 1.11 of the oracle's standard errors, the worst pair's median
    correlation gap at most 0.0118 and 0.0102. Held: 2.5 SE and 0.025 on
    every seed."""
    for seed in range(10):
        c = compare_with_margins(form, seed)
        assert np.max(np.abs(c["gap"])) <= 2.5, (form, seed, c)
        assert c["rho_gap"] <= 0.025, (form, seed, c["rho_gap"])
