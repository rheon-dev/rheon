"""``bocpd``'s pruned run-length posterior against the whole one that
``bayesian_changepoint_detection`` computes.

Every other ``bocpd`` oracle turns pruning off (``prune_below = 0``,
``max_run`` past the stream), but a user runs the defaults, ``prune_below =
1e-6`` and ``max_run = 10,000``. Here the model runs at them, and the
package -- Adams and MacKay's recursion with nothing pruned, at the
normal-inverse-gamma prior ``"diag"`` is at one feature -- gives the
posterior that pruning approximates.

Pruning drops whole runs, so what it keeps is the package's posterior up to
one known factor. Write ``L_c(r)`` for the model's joint over run length
``r`` after ``c`` rows (its state keeps it unnormalised, as the log joint
``P(r_c = r, x_1..c)``), ``J_c(r)`` for the package's, and ``Z_c`` for the
package's evidence ``P(x_1..c)``. A run of ``r >= 1`` rows grows from the
empty run of ``c - r`` rows before by the same factors on both sides, so a
kept run is the package's run scaled by what its birth kept:

    L_c(r) / J_c(r) = L_{c-r}(0) / J_{c-r}(0)

and the empty run is born of the whole posterior, ``L_c(0) = H |L_{c-1}|
exp(loglik_c)``. That identity holds for every kept run on every row; a run
whose statistics were moved to another run's place by a drop, or a drop of
the empty run, breaks it.

Every factor ``rho_s = L_s(0) / J_s(0)`` follows from the mass pruned:
the empty run is born of the whole posterior on both sides, so ``ln rho_s``
is ``ln(|L_{s-1}| / Z_{s-1}) + loglik_s - loglik*_s``, and

    |L_c| / Z_c = prod_s (1 - delta_s) exp(loglik_s - loglik*_s)

``delta_s`` the share of row ``s``'s posterior the prune dropped, each
dropped run under ``prune_below`` (so ``delta_s <= prune_below`` times the
runs dropped), ``loglik_s`` the model's row log predictive and
``loglik*_s`` the package's. Every kept entry is then at most ``kappa_c``
times the package's, ``kappa_c`` the largest ``rho`` of a kept run's birth,
and for nonnegative vectors ``a <= kappa b`` the total-variation distance
of their normalisations is at most ``1 - |a| / (kappa |b|)``. That is the
bound: ``p_change``, the posterior probability of ``r <= 1``, is within it
of the package's, and the posterior's mode, which dates the run's start
(``run_mode``), is the package's wherever the package's mode leads the next
run length by more than twice it. The bound is loose where the package
keeps much of its mass on lineages the prune cut (it reaches 1 on one of
the fifty streams, whose ``p_change`` moved 2.5e-5); it is a bound, not an
estimate.

``max_run`` is not a truncation: the runs that reach ``max_run - 1`` rows
fold into one, which then forgets as an exponentially weighted run, so its
predictive is no run's in the package and no bound from dropped mass holds
past it. What the fold must leave alone is held: every run below the cap,
by the identity above. At the default ``max_run`` the fold first acts after
9,999 rows of one regime; it is held at a cap of 30, where it acts on every
row once a regime is that long.

The robust emission (``robust_beta``) has no third-party implementation. It
weighs each row by ``(pi(x) / pi(mode))^beta`` under each run, in the
message the row passes and in what the run learns. Knoblauch, Jewson and
Damoulas' (2018) robust BOCPD, the nearest published method, is another
posterior: generalised Bayes under the beta-divergence loss ``-f(x)^beta /
beta + integral f^(1 + beta) / (1 + beta)``, whose parameter posterior is
not conjugate and which they approximate variationally, so it cannot hold
this one. The last test holds the robust emission to its own definition,
written from the model's documentation, with scipy's Student-t density in
place of the model's.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import bayesian_changepoint_detection.online_changepoint_detection as bcd
import numpy as np
import polars as pl
import pytest
from scipy.special import logsumexp
from scipy.stats import t as student_t

import ledger
from rheon import polars as rp

TIER = "mixed"

SEEDS = range(50)

#: The default ``prune_below`` the model runs at, read by the bound.
PRUNE_BELOW = 1e-6

#: The prior: the default hazard, the package's ``StudentT(alpha =
#: nu0/2, beta = covariance_prior/2, kappa = mean_precision_prior, mu =
#: mean_prior)`` at the default ``degrees_of_freedom_prior`` 3.
HAZARD = 250.0
KAPPA0 = 1.0
PSI0 = 1.0
NU0 = 3.0

#: The identity's error in nats, one log joint against the package's,
#: whose magnitude grows with the rows (about 1.4 a row). Measured
#: 2026-11-04 over ``SEEDS`` (aarch64, macOS): the worst 1.16e-11 at 600
#: rows. Each side sums ``n`` row log predictives, each rounded to a few
#: ulps of a number near 1, and the package normalises each column, so the
#: error grows with ``n u``; the bound leaves about ten times the worst.
IDENTITY_TOL = 1e-10

#: ``p_change`` against the package beyond the bound, absolute:
#: ``TestBocpdAtALevel``'s round-off at level 0.
P_TOL = 1e-9


def stream(seed: int, n: int = 600) -> np.ndarray:
    """Four mean shifts of 1.5 to 4 noise standard deviations at random rows:
    each collapses the posterior, which is where the prune drops the most."""
    rng = np.random.default_rng(seed)
    level = np.zeros(n)
    for cut in np.sort(rng.choice(np.arange(30, n - 30), 4, replace=False)):
        level[cut:] += rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 4.0)
    return level + rng.normal(0.0, 1.0, n)


def spec(**kw: Any) -> dict[str, Any]:
    return rp.spec.bocpd(
        "b",
        features=["x"],
        hazard_rows=HAZARD,
        covariance_type="diag",
        mean_prior=[0.0],
        mean_precision_prior=KAPPA0,
        covariance_prior=[PSI0],
        min_weight=0.0,
        **kw,
    )


def model_run(x: np.ndarray, **kw: Any) -> tuple[pl.DataFrame, list[dict[int, float]]]:
    """The model's rows, and after each its log joint over the kept run
    lengths, read from the state the bank writes as JSON (the model emits
    the posterior's mode and mean only)."""
    bank = rp.ModelBank([spec(**kw)])
    outs, joints = [], []
    for v in x:
        outs.append(bank.fit_predict(pl.DataFrame({"x": [v]})).unnest("b"))
        state = json.loads(bank.write_json())["states"][0][0][1]
        model = state["models"][0]["model"]["Bocpd"]
        joints.append(
            {int(run["len"]): lj for run, lj in zip(model["runs"], model["logjoint"], strict=True)}
        )
    return pl.concat(outs), joints


ledger.guard(
    "bocpd_pruning",
    lambda: [rp.ModelBank([spec()]).fit_predict(pl.DataFrame({"x": stream(s)})) for s in range(10)],
    oracles=["bayesian-changepoint-detection"],
    bits=52,
)


class Recording(bcd.StudentT):
    """The package's model, keeping each row's predictive densities, which
    its recursion uses and does not return."""

    def __init__(self, *args: float) -> None:
        super().__init__(*args)
        self.densities: list[np.ndarray] = []

    def pdf(self, data: float) -> np.ndarray:
        p = super().pdf(data)
        self.densities.append(p)
        return p


def package_run(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The package's posterior ``R[r, c]`` after ``c`` rows, its log evidence
    ``ln Z_c`` (``ln Z_0 = 0``), and each row's log predictive."""
    model = Recording(NU0 / 2.0, PSI0 / 2.0, KAPPA0, 0.0)
    r, _ = bcd.online_changepoint_detection(
        x, functools.partial(bcd.constant_hazard, HAZARD), model
    )
    loglik = np.log([r[: c + 1, c] @ model.densities[c] for c in range(len(x))])
    return r, np.concatenate([[0.0], np.cumsum(loglik)]), loglik


def kept_shares(
    delta: np.ndarray, loglik: np.ndarray, loglik_pkg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """From the dropped shares and the two predictives: ``ln(|L_c| / Z_c)``,
    the share of the package's evidence the model keeps after ``c`` rows
    (index 0 before any row), and ``ln rho_c = ln(L_c(0) / J_c(0))``, the
    share its empty run after ``c`` rows keeps. The empty run is born of the
    whole posterior on both sides, ``H |L_{c-1}| exp(loglik_c)`` against ``H
    Z_{c-1} exp(loglik*_c)``, so ``ln rho_c`` is the share after ``c - 1``
    rows plus ``loglik_c - loglik*_c``; the prune then keeps ``1 - delta_c``
    of the model's."""
    before = np.concatenate([[0.0], np.cumsum(np.log1p(-delta[:-1]) + (loglik - loglik_pkg)[:-1])])
    rho = np.concatenate([[0.0], before + loglik - loglik_pkg])
    whole = np.concatenate([[0.0], before + loglik - loglik_pkg + np.log1p(-delta)])
    return whole, rho


def identity_error(
    joints: list[dict[int, float]],
    r: np.ndarray,
    log_z: np.ndarray,
    rho: np.ndarray,
    below: int | None = None,
) -> float:
    """The worst ``|ln L_c(r) - ln J_c(r) - ln rho_{c-r}|`` over every kept
    run (shorter than ``below`` where given)."""
    worst = 0.0
    for c, joint in enumerate(joints, start=1):
        for length, lj in joint.items():
            if below is not None and length >= below:
                continue
            g = lj - np.log(r[length, c]) - log_z[c]
            worst = max(worst, abs(g - rho[c - length]))
    return worst


def dropped_shares(
    out: pl.DataFrame, joints: list[dict[int, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """The share of each row's posterior the prune dropped -- the joint after
    the row against the one before its prune, ``|L_{c-1}| exp(loglik_c)`` --
    and how many runs it dropped."""
    total = np.array([logsumexp(list(j.values())) for j in joints])
    kept = total - np.concatenate([[0.0], total[:-1]]) - out["loglik"].to_numpy()
    runs = np.array([len(j) for j in joints])
    return -np.expm1(kept), np.concatenate([[1], runs[:-1]]) + 1 - runs


def pruning_check(seed: int) -> dict[str, float]:
    """The model at its default pruning against the package on one stream:
    the identity's error, the dropped shares against the rule, ``p_change``
    beyond the bound, and ``run_mode`` where the bound decides it."""
    x = stream(seed)
    n = len(x)
    out, joints = model_run(x)
    r, log_z, loglik_pkg = package_run(x)
    delta, dropped = dropped_shares(out, joints)
    whole, rho = kept_shares(delta, out["loglik"].to_numpy(), loglik_pkg)
    # The bound after each row. Every kept run is at most kappa_c times the
    # package's, kappa_c the largest rho of a kept run's birth, so the
    # distance is at most 1 - |L_c| / (kappa_c Z_c). Against kappa_c, not 1:
    # a share lost long ago scales every later birth alike and moves no
    # posterior.
    kappa = np.array([max(rho[c - length] for length in j) for c, j in enumerate(joints, 1)])
    bound = np.maximum(-np.expm1(whole[1:] - kappa), 0.0)
    p_model = out["p_change"].to_numpy()
    p_pkg = r[0, 1:] + r[1, 1:]
    # run_mode is the mode before the row: R[:, t] and the bound after
    # t rows.
    top2 = np.sort(r[:, :n], axis=0)[-2:]
    before = np.concatenate([[0.0], bound[:-1]])
    decided = top2[1] - top2[0] > 2.0 * before + 2.0 * P_TOL
    modes = r[:, :n].argmax(axis=0)
    return {
        "identity": identity_error(joints, r, log_z, rho),
        "rule": float(np.max(delta - PRUNE_BELOW * dropped)),
        "least_delta": float(delta.min()),
        "dropped": float(dropped.sum()),
        "largest_bound": float(bound.max()),
        "p_excess": float(np.max(np.abs(p_model - p_pkg) - bound)),
        "p_moved": float(np.abs(p_model - p_pkg).max()),
        "decided": float(decided.sum() / n),
        "mode_misses": float((out["run_mode"].to_numpy()[decided] != modes[decided]).sum()),
    }


def assert_pruning(m: dict[str, float]) -> None:
    # Evidence the prune binds: runs dropped, and p_change moved off the
    # package's by more than round-off.
    assert m["dropped"] > 100 and m["p_moved"] > 1e-8, m
    assert m["identity"] <= IDENTITY_TOL, m
    # Each dropped run held under prune_below: the shares by the rule, to the
    # round-off of a difference of two log-sums near 1.4 n.
    assert m["rule"] <= 1e-11 and m["least_delta"] >= -1e-11, m
    assert m["p_excess"] <= P_TOL, m
    assert m["decided"] >= 0.5 and m["mode_misses"] == 0, m


class TestDefaultPruningIsThePackagesPosterior:
    """At the default ``prune_below`` and ``max_run``, the kept runs are the
    package's up to their births' kept shares, and ``p_change`` and
    ``run_mode`` are within the bound the dropped mass gives."""

    def test_one_seed(self):
        m = pruning_check(0)
        print({k: f"{v:.3g}" for k, v in m.items()})
        assert_pruning(m)

    @pytest.mark.oracle_grid(reason="50 seeds of 600 rows, the state read after every row")
    def test_fifty_seeds(self):
        rows = [pruning_check(s) for s in SEEDS]
        for key in rows[0]:
            values = [r[key] for r in rows]
            print(f"{key}: worst {max(values):.3g}, least {min(values):.3g}")
        ledger.record("bocpd_pruning", "identity_nats", [r["identity"] for r in rows], floor=2e-11)
        ledger.record("bocpd_pruning", "mode_misses", [r["mode_misses"] for r in rows], exact=True)
        for m in rows:
            assert_pruning(m)


def test_the_fold_leaves_every_run_below_the_cap_the_packages():
    """At ``max_run = 30`` the fold acts on every row of a regime past 29
    rows. Every run below the cap is still the package's up to its birth's
    kept share; the folded run is not, and ``p_change`` moves by far more
    than any pruning bound, which is why only the runs below it are held."""
    x = stream(0)
    out, joints = model_run(x, max_run=30)
    r, log_z, loglik_pkg = package_run(x)
    delta, _ = dropped_shares(out, joints)
    _, rho = kept_shares(delta, out["loglik"].to_numpy(), loglik_pkg)
    folded = sum(1 for j in joints if 29 in j)
    moved = float(np.abs(out["p_change"].to_numpy() - (r[0, 1:] + r[1, 1:])).max())
    error = identity_error(joints, r, log_z, rho, below=29)
    print(f"rows with a folded run {folded}, p_change moved {moved:.3f}, identity {error:.2e}")
    assert folded > 300 and moved > 0.05
    assert error <= IDENTITY_TOL


# --- a hazard on the clock, through the package's hazard function --------------------


def package_on_steps(x: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The package's posterior ``R``, its modes and each row's log predictive,
    with ``after[c]`` the chance of a break after row ``c``. The package takes
    no vector of hazards: it calls its hazard function once a row, on that
    row's run lengths, so a function that counts its calls gives each row
    its own chance."""
    calls = iter(range(len(x)))

    def hazard_function(r: np.ndarray) -> np.ndarray:
        return np.full(r.shape, after[next(calls)])

    model = Recording(NU0 / 2.0, PSI0 / 2.0, KAPPA0, 0.0)
    r, maxes = bcd.online_changepoint_detection(x, hazard_function, model)
    loglik = np.log([r[: c + 1, c] @ model.densities[c] for c in range(len(x))])
    return r, maxes, loglik


def test_a_clock_hazard_on_irregular_steps_is_the_packages_with_each_steps_chance():
    """``hazard = tau`` on a number clock puts the chance ``h = 1 - exp(-dt /
    tau)`` of the step into a row before the row (task 547). The package
    applies its hazard after the row it is called on, so a hazard function
    that returns the *next* step's chance makes its ``R[:, c]``, the
    posterior before row ``c``, the clock form's: the model's ``run_mode``
    and ``run_mean`` read it and its ``loglik`` is that posterior's
    predictive. The clock form's ``p_change`` is the chance row ``c`` began
    a run with no break after it yet, ``R[1, c + 1] / (1 - h_{c + 1})``,
    whatever the chance the function gave after the last row. Irregular
    steps in seconds, shared stamps among them (no chance at all), nothing
    pruned, and a level that moves twice."""
    rng = np.random.default_rng(547)
    n, tau = 300, 40.0
    gaps = rng.exponential(10.0, n).round(1)
    gaps[0] = 0.0
    gaps[rng.random(n) < 0.05] = 0.0
    secs = np.cumsum(gaps)
    level = np.where(np.arange(n) < 100, 0.0, np.where(np.arange(n) < 220, 3.0, -1.0))
    x = level + rng.normal(0.0, 1.0, n)
    # The chance of the step after row c, and an arbitrary one after the last.
    after = np.concatenate([-np.expm1(-np.diff(secs) / tau), [0.5]])
    r, maxes, loglik = package_on_steps(x, after)
    clock = rp.spec.bocpd(
        "b",
        features=["x"],
        hazard=tau,
        clock="t",
        gap_cap=1e9,
        covariance_type="diag",
        mean_prior=[0.0],
        mean_precision_prior=KAPPA0,
        covariance_prior=[PSI0],
        prune_below=0.0,
        max_run=n + 2,
        min_weight=0.0,
    )
    out = rp.ModelBank([clock]).fit_predict(pl.DataFrame({"t": secs, "x": x})).unnest("b")
    # Tolerances: ``TestBocpdAtALevel``'s at level 0, one recursion against
    # another through libm.
    np.testing.assert_allclose(out["p_change"], r[1, 1:] / (1.0 - after), rtol=0, atol=P_TOL)
    np.testing.assert_allclose(out["run_mean"], np.arange(n + 1) @ r[:, :n], rtol=1e-9)
    np.testing.assert_allclose(out["loglik"], loglik, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(out["run_mode"], maxes[:n])
    # The steps acted: a stamp shared with the row before begins no run.
    shared = np.flatnonzero(gaps[1:] == 0.0) + 1
    assert len(shared) > 5 and (out["p_change"].to_numpy()[shared] == 0.0).all()
    assert out["p_change"].max() > 0.5


#: The clock form's gap cap, its ``tau`` and the steps the test below puts
#: at the cap and beside it, in clock units.
GAP_CAP = 30.0
CAP_TAU = 40.0
AT_THE_CAP = (29.9, 30.0, 30.1, 30.0 + 1e-9, 45.0, 120.0, 600.0)


def test_a_clock_hazard_is_the_packages_on_both_sides_of_the_gap_cap():
    """A step past ``gap_cap`` counts as the cap, so the chance it carries is
    ``1 - exp(-min(d, cap) / tau)``. The package, fed each step's chance so
    capped, is the clock form's posterior on every learned row, with steps
    below the cap, at it, a hair past it and far past it, and shared stamps.
    A row of weight 0 counts in no clock: its step goes to the next learned
    row, so a gap it splits into two pieces below the cap is one step past
    it, capped once, and its wild value is learned by no run.

    Tolerances: ``TestBocpdAtALevel``'s at level 0, as the test above, one
    recursion against another through libm. Measured 2026-11-24 (aarch64,
    Linux): ``p_change`` 2.8e-14, ``run_mean`` 5.1e-14 relative, ``loglik``
    4.5e-14, ``run_mode`` exact. The package fed the uncapped chances instead
    misses ``p_change`` by 0.82, so the bound sees the cap."""
    rng = np.random.default_rng(5472)
    n = 300
    gaps = rng.exponential(10.0, n).round(1)
    gaps[0] = 0.0
    gaps[rng.random(n) < 0.05] = 0.0
    far = rng.choice(np.arange(1, n), 30, replace=False)
    gaps[far] = rng.uniform(GAP_CAP, 8.0 * GAP_CAP, len(far)).round(1)
    near = rng.choice(np.setdiff1d(np.arange(1, n), far), len(AT_THE_CAP), replace=False)
    gaps[near] = AT_THE_CAP
    # Three steps of 50, each split into two pieces of 25 by a row of
    # weight 0 forty noise sds out.
    split = rng.choice(np.setdiff1d(np.arange(1, n), np.concatenate([far, near])), 3, replace=False)
    gaps[split] = 50.0
    secs = np.cumsum(gaps)
    level = np.where(np.arange(n) < 100, 0.0, np.where(np.arange(n) < 220, 3.0, -1.0))
    x = level + rng.normal(0.0, 1.0, n)
    frame = pl.DataFrame({"t": secs, "x": x, "w": np.ones(n)})
    weightless = pl.DataFrame({"t": secs[split] - 25.0, "x": 40.0, "w": 0.0})
    stream = pl.concat([frame, weightless]).sort("t", maintain_order=True)
    learned = stream["w"].to_numpy() > 0
    steps = np.diff(secs)
    assert learned.sum() == n and (stream["x"].filter(pl.Series(~learned)) == 40.0).all()
    # Both sides of the cap, and within a hair of it on both.
    assert (steps > GAP_CAP).sum() >= 35 and ((steps > 0) & (steps < GAP_CAP)).sum() > 200
    assert (np.abs(steps - GAP_CAP) < 0.2).sum() >= 4
    assert (np.abs(steps[split - 1] - 50.0) < 1e-9).all()
    capped = -np.expm1(-np.minimum(steps, GAP_CAP) / CAP_TAU)
    # The chance of the step after each learned row, an arbitrary one after
    # the last.
    after = np.concatenate([capped, [0.5]])
    r, maxes, loglik = package_on_steps(x, after)
    spec = rp.spec.bocpd(
        "b",
        features=["x"],
        hazard=CAP_TAU,
        clock="t",
        gap_cap=GAP_CAP,
        weight="w",
        covariance_type="diag",
        mean_prior=[0.0],
        mean_precision_prior=KAPPA0,
        covariance_prior=[PSI0],
        prune_below=0.0,
        max_run=n + 2,
        min_weight=0.0,
    )
    out = rp.ModelBank([spec]).fit_predict(stream).unnest("b").filter(pl.Series(learned))
    np.testing.assert_allclose(out["p_change"], r[1, 1:] / (1.0 - after), rtol=0, atol=P_TOL)
    np.testing.assert_allclose(out["run_mean"], np.arange(n + 1) @ r[:, :n], rtol=1e-9)
    np.testing.assert_allclose(out["loglik"], loglik, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(out["run_mode"], maxes[:n])
    # The cap acted: the package on the uncapped steps is another posterior.
    uncapped = np.concatenate([-np.expm1(-steps / CAP_TAU), [0.5]])
    r_far, _, _ = package_on_steps(x, uncapped)
    missed = np.abs(out["p_change"].to_numpy() - r_far[1, 1:] / (1.0 - uncapped)).max()
    print(f"p_change against the uncapped chances: {missed:.3g}")
    assert missed > 1e-3


# --- the robust emission against its definition ---------------------------------------

#: The robust emission's prior: two features, each the normal-inverse-gamma
#: at these values.
ROBUST_HAZARD = 100.0
ROBUST_NU0 = 3.0


def robust_stream(seed: int, n: int = 200) -> np.ndarray:
    """Two features; their means move by (3, -2) noise sds at row 100; and
    two rows 15 sds out on the first feature, one in each regime: the rows
    the robust weight is for."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    x[n // 2 :] += [3.0, -2.0]
    x[[n // 5, 3 * n // 4], 0] += 15.0
    return x


def robust_model(x: np.ndarray, beta: float | None) -> pl.DataFrame:
    """The robust emission at ``beta``, or the plain one at ``None``."""
    features = [f"x{i}" for i in range(x.shape[1])]
    spec = rp.spec.bocpd(
        "b",
        features=features,
        hazard_rows=ROBUST_HAZARD,
        covariance_type="diag",
        robust_beta=beta,
        mean_prior=[0.0] * x.shape[1],
        mean_precision_prior=KAPPA0,
        covariance_prior=[PSI0],
        degrees_of_freedom_prior=ROBUST_NU0,
        prune_below=0.0,
        max_run=len(x) + 2,
        min_weight=0.0,
    )
    frame = pl.DataFrame({f: x[:, i] for i, f in enumerate(features)})
    return rp.ModelBank([spec]).fit_predict(frame).unnest("b")


def robust_definition(x: np.ndarray, beta: float) -> dict[str, np.ndarray]:
    """The β-power weighted recursion the model states, from its definition,
    with scipy's Student-t density.

    Run ``r`` holds its rows at weights ``w``: ``W = sum w``, the weighted
    mean ``m`` and scatter ``S`` of each feature. Its predictive, per
    feature, is the normal-inverse-gamma posterior's Student-t at ``W``
    pseudo-rows: ``nu = nu0 + W``, ``kappa = kappa0 + W``, location
    ``(kappa0 mu0 + W m) / kappa`` and squared scale ``psi (kappa + 1) /
    (nu kappa)``, ``psi = psi0 + S + kappa0 W (m - mu0)^2 / kappa``. A row
    ``x`` gets the weight ``w_r = min(1, (pi_r(x) / pi_r(mode))^beta)`` under
    each run, the density at ``x`` over the density at the predictive's
    mode, and passes the tempered message ``pi_r(x)^(w_r)``: growth
    ``J(r + 1) = J(r) pi^w (1 - H)``, the empty run ``sum J pi^w H``. Each
    run then learns the row at its ``w_r``, by the weighted Welford step.
    ``p_change`` is the posterior of ``r <= 1`` after the row; ``run_mode``,
    ``run_mean`` and ``loglik`` (``ln sum p pi``, untempered) are read from
    the posterior before it."""
    n, d = x.shape
    h = 1.0 / ROBUST_HAZARD
    weight = np.zeros(1)
    mean = np.zeros((1, d))
    scatter = np.zeros((1, d))
    length = np.zeros(1)
    joint = np.zeros(1)
    out = {k: np.zeros(n) for k in ("p_change", "run_mode", "run_mean", "loglik")}
    for t, row in enumerate(x):
        kappa = KAPPA0 + weight[:, None]
        nu = ROBUST_NU0 + weight[:, None]
        loc = weight[:, None] * mean / kappa
        psi = PSI0 + scatter + KAPPA0 * weight[:, None] / kappa * mean**2
        scale = np.sqrt(psi * (kappa + 1.0) / (nu * kappa))
        density = student_t.logpdf(row, df=nu, loc=loc, scale=scale).sum(axis=1)
        at_mode = student_t.logpdf(loc, df=nu, loc=loc, scale=scale).sum(axis=1)
        w = np.minimum(1.0, np.exp(beta * (density - at_mode)))
        before = np.exp(joint - logsumexp(joint))
        out["run_mode"][t] = length[np.argmax(before)]
        out["run_mean"][t] = before @ length
        out["loglik"][t] = logsumexp(np.log(before) + density)
        base = joint + w * density
        joint = np.concatenate([[logsumexp(base) + np.log(h)], base + np.log1p(-h)])
        out["p_change"][t] = np.exp(logsumexp(joint[:2]) - logsumexp(joint))
        grown = weight + w
        dev = row - mean
        moved = mean + (w / np.where(grown > 0, grown, 1.0))[:, None] * dev
        scatter = np.vstack([np.zeros(d), scatter + w[:, None] * dev * (row - moved)])
        mean = np.vstack([np.zeros(d), moved])
        weight = np.concatenate([[0.0], grown])
        length = np.concatenate([[0.0], length + 1.0])
    return out


#: Absolute on ``p_change``, relative to ``1 + |x|`` on ``run_mean`` and
#: ``loglik``. Measured 2026-11-04 over the seeds and betas below (aarch64,
#: macOS): the worst 1.4e-13 (a ``run_mean`` near 100). Each side sums the
#: same few hundred log densities through its own log-sum-exp.
ROBUST_TOL = 1e-11


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.05, 0.1, 0.3])
def test_the_robust_emission_is_its_definition(beta, seed):
    """``robust_beta`` at 0.05 and 0.1, which find the shift, and 0.3, which
    the module documents as forgiving whole regimes: each is the β-power
    weighted recursion written from its definition, row for row."""
    x = robust_stream(seed)
    out = robust_model(x, beta)
    want = robust_definition(x, beta)
    p_err = float(np.abs(out["p_change"].to_numpy() - want["p_change"]).max())
    errors = {
        "p_change": p_err,
        **{
            k: float((np.abs(out[k].to_numpy() - want[k]) / (1.0 + np.abs(want[k]))).max())
            for k in ("run_mean", "loglik")
        },
    }
    print({k: f"{v:.2g}" for k, v in errors.items()})
    np.testing.assert_array_equal(out["run_mode"].to_numpy(), want["run_mode"])
    # Evidence the weight acted: the plain emission reads the first 15-sd
    # row as a break, the robust one does not.
    outlier = len(x) // 5
    plain = robust_model(x, None)["p_change"][outlier]
    print(f"p_change at the outlier: plain {plain:.3f}, robust {out['p_change'][outlier]:.3g}")
    assert plain > 0.5 > out["p_change"][outlier]
    assert max(errors.values()) <= ROBUST_TOL, errors
