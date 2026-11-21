"""``hawkes`` against tick's least-squares Hawkes models (task 498).

tick's ``ModelHawkesSumExpKernLeastSq`` (and ``ModelHawkesExpKernLeastSq``,
its one-decay form) computes the contrast ``hawkes`` minimises,
``Σ_i ∫ λ_i² dt - 2 ∫ λ_i dN_i``, divided by the number of events, for
coefficients laid out ``[μ_1 .. μ_D, α (D, D, U) raveled]`` -- per node
``i`` the row ``hawkes`` reports in ``coef``, ``μ_i`` then ``α_iju`` with the
source ``j`` outer. The division moves no minimiser. Neither model has a
Hessian for the sum of exponentials, so its exact minimiser is read from
its own gradient: the loss is quadratic, so ``H e_c = ∇(e_c) - ∇(0)`` and
the minimiser is ``-H⁻¹ ∇(0)``.

The streams are tick's own simulation, seeded; the bank reads them as rows
of a numeric clock, one row an event (the nodes' clocks never meet), a row
at 0 before them so the observation starts where tick's does, and a row at
the end time with no event, where the last fit is read. tick takes no
weights, so the forgotten fit is held to its definition in
``tests/test_hawkes.py`` instead.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from tick.hawkes import (
    ModelHawkesExpKernLeastSq,
    ModelHawkesSumExpKernLeastSq,
    SimuHawkesSumExpKernels,
)

from rheon import polars as rp

TIER = "essential"

#: Where tick's minimiser and this fit part: relative to the largest
#: coefficient. Measured at most 3.9e-14 over the cases below (the window 1.1e-14).
TOL = 1e-12


def _simulate(decays: list[float], d: int, end: float, seed: int) -> list[np.ndarray]:
    """``d`` nodes excited through ``decays``, by tick's own simulator: a
    baseline of 0.3 to 0.5 and kernels whose norms sum to 0.6 a node."""
    u = len(decays)
    rng = np.random.default_rng(seed)
    adjacency = rng.uniform(0.2, 1.0, size=(d, d, u))
    adjacency *= 0.6 / adjacency.sum(axis=(1, 2), keepdims=True)
    sim = SimuHawkesSumExpKernels(
        adjacency=adjacency,
        decays=np.array(decays),
        baseline=list(np.linspace(0.3, 0.5, d)),
        end_time=end,
        seed=seed,
        verbose=False,
    )
    sim.simulate()
    return [np.asarray(e, dtype=float) for e in sim.timestamps]


def _frame(events: list[np.ndarray], end: float) -> pl.DataFrame:
    """One row per event, its node's column 1 and the others 0, between a
    row at 0 and a row at ``end`` with no event."""
    d = len(events)
    times = np.concatenate(events)
    nodes = np.concatenate([np.full(len(e), j) for j, e in enumerate(events)])
    order = np.argsort(times, kind="stable")
    times, nodes = times[order], nodes[order]
    t = np.concatenate([[0.0], times, [end]])
    cols = {"t": t}
    for j in range(d):
        cols[f"n{j}"] = np.concatenate([[0.0], (nodes == j).astype(float), [0.0]])
    return pl.DataFrame(cols)


def _spec(decays: list[float], d: int, **kw: object) -> dict:
    return rp.spec.hawkes(
        "h",
        targets=[f"n{j}" for j in range(d)],
        kernel_decays=decays,
        clock="t",
        gap_cap=1e12,
        half_life=float("inf"),
        **kw,
    )


def _ticks_minimiser(model) -> np.ndarray:
    """tick's exact minimiser, from its gradient, per node ``[μ_i, α_i..]``."""
    n = model.n_coeffs
    g0 = model.grad(np.zeros(n))
    h = np.column_stack([model.grad(np.eye(n)[c]) - g0 for c in range(n)])
    x = -np.linalg.solve(h, g0)
    d = model.n_nodes
    mu, alpha = x[:d], x[d:].reshape(d, -1)
    return np.concatenate([np.concatenate([[mu[i]], alpha[i]]) for i in range(d)])


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


CASES = [
    ([1.0, 5.0], 1, 2_000.0, 1),
    ([0.5, 3.0, 20.0], 1, 2_000.0, 2),
    ([1.0, 8.0], 2, 1_500.0, 3),
    ([0.7, 4.0, 15.0], 2, 1_200.0, 4),
    ([1.5, 6.0], 3, 800.0, 5),
]


@pytest.mark.parametrize(("decays", "d", "end", "seed"), CASES)
def test_the_fit_is_ticks_minimiser(decays, d, end, seed):
    """The fit after the last row, read at the end time, is the minimiser
    of tick's contrast over the same events and observation."""
    events = _simulate(decays, d, end, seed)
    bank = rp.ModelBank([_spec(decays, d)])
    bank.fit(_frame(events, end))
    got = bank.coef()["coef"].to_numpy()
    model = ModelHawkesSumExpKernLeastSq(decays=np.array(decays))
    model.fit(events, end_times=end)
    want = _ticks_minimiser(model)
    assert _rel(got, want) < TOL, (_rel(got, want), got, want)


def test_one_decay_is_ticks_exp_kernel():
    """With one decay shared by every pair, the fit is tick's
    ``ModelHawkesExpKernLeastSq``'s minimiser too."""
    events = _simulate([2.0], 2, 1_500.0, 6)
    bank = rp.ModelBank([_spec([2.0], 2)])
    bank.fit(_frame(events, 1_500.0))
    got = bank.coef()["coef"].to_numpy()
    model = ModelHawkesExpKernLeastSq(decays=2.0)
    model.fit(events, end_times=1_500.0)
    want = _ticks_minimiser(model)
    assert _rel(got, want) < TOL, (_rel(got, want), got, want)


def test_the_loss_is_ticks_scaled_by_the_events():
    """tick's loss is the contrast divided by the number of events: at the
    minimiser the contrast is ``-θᵀ b``, and ``b``'s count part times ``μ``
    plus its excitation part times ``α`` is ``Σ_k λ(t_k−)`` at that ``θ``;
    so tick's loss at this fit, times ``n_jumps``, is minus the fitted
    intensities summed over the events."""
    decays, d, end = [1.0, 6.0], 2, 1_000.0
    events = _simulate(decays, d, end, 8)
    bank = rp.ModelBank([_spec(decays, d)])
    bank.fit(_frame(events, end))
    theta = bank.coef()["coef"].to_numpy().reshape(d, -1)
    model = ModelHawkesSumExpKernLeastSq(decays=np.array(decays))
    model.fit(events, end_times=end)
    x = np.concatenate([theta[:, 0], theta[:, 1:].ravel()])
    total = 0.0
    for i, ev in enumerate(events):
        for t in ev:
            phi = [1.0]
            for src in events:
                before = src[src < t]
                phi.extend(b * np.exp(-b * (t - before)).sum() for b in decays)
            total += float(np.dot(theta[i], phi))
    got, want = model.loss(x) * model.n_jumps, -total
    assert abs(got - want) <= 1e-11 * abs(want), (got, want)


def test_each_rows_intensity_is_ticks_fit_on_the_rows_before():
    """A row's intensity is the minimiser of tick's contrast over the rows
    before it alone, observed to the row's clock, times the excitation
    before the row's events: out of sample by construction."""
    decays, d, end = [1.0, 5.0], 2, 600.0
    events = _simulate(decays, d, end, 9)
    df = _frame(events, end)
    out = rp.ModelBank([_spec(decays, d)]).fit_predict(df).unnest("h")
    t = df["t"].to_numpy()
    for row in range(len(df) // 4, len(df), len(df) // 7):
        now = t[row]
        before = [e[e < now] for e in events]
        model = ModelHawkesSumExpKernLeastSq(decays=np.array(decays))
        model.fit(before, end_times=now)
        theta = _ticks_minimiser(model).reshape(d, -1)
        phi = [1.0]
        for src in before:
            phi.extend(b * np.exp(-b * (now - src)).sum() for b in decays)
        for i in range(d):
            got = out[f"intensity_n{i}"][row]
            want = float(np.dot(theta[i], phi))
            assert abs(got - want) <= 1e-10 * (1.0 + abs(want)), (row, i, got, want)


def test_the_window_is_tick_refit_on_its_events():
    """Under ``window_events = W`` the fit at a row is tick's minimiser
    over the last ``W`` events, shifted to the clock of the event dropped
    last and observed to the row's clock, across many turnovers."""
    decays, d, end, w = [1.0, 6.0], 2, 1_000.0, 150
    events = _simulate(decays, d, end, 10)
    df = _frame(events, end)
    out = rp.ModelBank([_spec(decays, d, window_events=w, coef_every=0)]).fit_predict(df)
    coef = out["h"].struct.field("coef")
    t = df["t"].to_numpy()
    event_rows = np.flatnonzero(df.select(pl.sum_horizontal(pl.exclude("t")) > 0).to_series())
    worst = 0.0
    for row in range(3 * w, len(df), 97):
        kept = event_rows[event_rows <= row][-w:]
        dropped = event_rows[event_rows < kept[0]]
        start = t[dropped[-1]] if len(dropped) else 0.0
        window = [np.sort(t[kept][df[f"n{j}"].to_numpy()[kept] > 0]) - start for j in range(d)]
        model = ModelHawkesSumExpKernLeastSq(decays=np.array(decays))
        model.fit(window, end_times=t[row] - start)
        want = _ticks_minimiser(model)
        got = np.asarray(coef[row])
        worst = max(worst, _rel(got, want))
    assert worst < TOL, worst
