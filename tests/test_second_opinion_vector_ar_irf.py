"""``vector_ar``'s impulse responses and variance decomposition against
statsmodels' ``VARResults.irf`` and ``.fevd`` across its shapes.

``tests/test_vector_ar.py`` holds one shape (three series, order 2, a
constant, horizon 8) to ``irf``, ``orth_irf``, their standard errors and
``fevd``. Here the same reads are held across the shapes the fit can take:
one lag and several, two series to five, an exogenous regressor in every
equation, no constant (``trend="n"``), an order chosen by ``select``, and a
horizon of 20; and the decomposition's two exact facts on both sides of
its first horizon: at horizon 1 the first series' error is all its own
shock (the Cholesky factor is lower triangular), and from horizon 2 a
cross effect gives it a share of the others'.

**The bound.** Both sides solve the same least squares, the bank from its
moments by Cholesky and statsmodels from the design by its own solve, so
their coefficients part by at most ``forward_bound(κ, η)`` (Higham 7.2),
``κ`` the condition of the lag design's cross-product, ``Z'Z``, read here
with numpy, and ``η`` a few units of ``u`` for each of the ``d`` terms in
a moment: ``2κη/(1 − κη)`` with ``η = 4 d u``. A response ``Φ_i`` is a sum
of products of ``i`` lag matrices, so a relative error ``ε`` in them moves
it by about ``i ε``: the responses carry ``(H + 1)`` times the bound, the
orthogonalized ones and the shares that again through ``Σ_u``'s
Cholesky, and the standard errors that again through ``(Z'Z)⁻¹``. Each is
held relative to its own largest entry. Measured 2026-11-24 (x86-64,
Linux): at most 1.7e-3 of the bound.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from statsmodels.tsa.api import VAR

from rheon import polars as rp

TIER = "essential"
INF = float("inf")
U = 2.0**-53


def _var(n: int, k: int, p: int, seed: int, *, exog: bool) -> pl.DataFrame:
    """A stable VAR(p) of ``k`` series, each driven by its own lags and the
    next series' first lag, and with ``exog`` a regressor in every
    equation, after 300 rows of burn-in."""
    rng = np.random.default_rng([495, seed, k, p])
    a = np.zeros((p, k, k))
    a[0] = 0.4 * np.eye(k)
    for i in range(k - 1):
        a[0, i, i + 1] = 0.25
    if p > 1:
        a[1] = -0.15 * np.eye(k)
    if p > 2:
        a[2, k - 1, 0] = 0.1
    c = np.linspace(-1.0, 1.0, k)
    b = np.linspace(0.5, -0.5, k)
    m = n + 300
    x = rng.standard_normal(m)
    y = np.zeros((m, k))
    for t in range(p, m):
        y[t] = c + rng.standard_normal(k) + (b * x[t] if exog else 0.0)
        for j in range(p):
            y[t] += a[j] @ y[t - 1 - j]
    cols = {f"y{i}": y[300:, i] for i in range(k)}
    if exog:
        cols["x"] = x[300:]
    return pl.DataFrame(cols)


def _rel(got, want) -> float:
    got, want = np.asarray(got, float), np.asarray(want, float)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def _bound(z: np.ndarray) -> float:
    """``forward_bound(κ, η)`` for the normal equations of the design ``z``."""
    kappa = float(np.linalg.cond(z.T @ z))
    eta = 4 * z.shape[1] * U
    assert kappa * eta < 0.5
    return 2 * kappa * eta / (1 - kappa * eta)


def _design(ys: np.ndarray, p: int, x: np.ndarray | None, const: bool) -> np.ndarray:
    n = len(ys)
    cols = ([np.ones(n - p)] if const else []) + ([x[p:]] if x is not None else [])
    return np.column_stack(cols + [ys[p - 1 - j : n - 1 - j] for j in range(p)])


def _cube(frame: pl.DataFrame, col: str, k: int) -> np.ndarray:
    frame = frame.sort("period", "response", "impulse")
    return frame[col].to_numpy().reshape(-1, k, k)


CASES = {
    "two series, one lag": dict(k=2, p=1, exog=False, const=True, h=12),
    "four series, three lags, exog": dict(k=4, p=3, exog=True, const=True, h=12),
    "three series, no constant": dict(k=3, p=2, exog=False, const=False, h=12),
    "five series, horizon 20": dict(k=5, p=1, exog=False, const=True, h=20),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_reads_are_varresults_across_its_shapes(case):
    """``irf``, ``orth_irf``, both standard errors and ``fevd`` are
    ``VARResults``' on the same rows, each within its multiple of the
    solve's forward bound (the module doc)."""
    c = CASES[case]
    k, p, h = c["k"], c["p"], c["h"]
    df = _var(1500, k, p, 3, exog=c["exog"])
    names = [f"y{i}" for i in range(k)]
    spec = rp.spec.vector_ar(
        "v", targets=names, lags=p, features=["x"] if c["exog"] else None,
        fit_intercept=c["const"], half_life=INF,
    )  # fmt: skip
    bank = rp.ModelBank([spec])
    bank.fit(df)
    ys = df.select(names).to_numpy()
    x = df["x"].to_numpy() if c["exog"] else None
    res = VAR(ys, exog=None if x is None else x[:, None]).fit(p, trend="c" if c["const"] else "n")
    irf = res.irf(h)
    b = _bound(_design(ys, p, x, c["const"]))
    plain = bank.vector_ar(table="irf", periods=h, stderr=True)
    orth = bank.vector_ar(table="orth_irf", periods=h, stderr=True)
    assert _rel(_cube(plain, "value", k), irf.irfs) <= (h + 1) * b
    assert _rel(_cube(orth, "value", k), irf.orth_irfs) <= 2 * (h + 1) * b
    assert _rel(_cube(plain, "stderr", k), irf.stderr(orth=False)) <= 3 * (h + 1) * b
    assert _rel(_cube(orth, "stderr", k), irf.stderr(orth=True)) <= 4 * (h + 1) * b
    fevd = bank.vector_ar(table="fevd", periods=h).sort("response", "horizon", "shock")
    got = fevd["share"].to_numpy().reshape(k, h, k)
    want = res.fevd(h).decomp
    assert _rel(got, want) <= 2 * (h + 1) * b
    # Period 0's response is the identity, exactly, and the shares at
    # each horizon sum to 1.
    assert np.array_equal(_cube(plain, "value", k)[0], np.eye(k))
    np.testing.assert_allclose(got.sum(axis=2), 1.0, rtol=0, atol=k * 8 * U)


def test_the_first_horizons_decomposition_is_the_cholesky_ordering():
    """At horizon 1 the forecast error is the row's own shock: the first
    series' share of its own is 1 and of every other 0, exactly as
    statsmodels has it, since ``Θ_0 = P`` is lower triangular; the last
    series takes a share, however small, of every shock before it. At horizon 2 the first
    series, driven by the second's first lag, takes a share of the
    second's shock: the other side of the first horizon."""
    k, p, h = 3, 2, 4
    df = _var(1200, k, p, 5, exog=False)
    names = [f"y{i}" for i in range(k)]
    bank = rp.ModelBank([rp.spec.vector_ar("v", targets=names, lags=p, half_life=INF)])
    bank.fit(df)
    fevd = bank.vector_ar(table="fevd", periods=h).sort("response", "horizon", "shock")
    got = fevd["share"].to_numpy().reshape(k, h, k)
    want = VAR(df.select(names).to_numpy()).fit(p).fevd(h).decomp
    assert got[0, 0, 0] == pytest.approx(1.0, abs=8 * U) and want[0, 0, 0] == pytest.approx(1.0)
    assert np.all(np.abs(got[0, 0, 1:]) <= 8 * U) and np.all(np.abs(want[0, 0, 1:]) <= 8 * U)
    assert np.all(got[k - 1, 0, :] > 0)
    assert got[0, 1, 1] > 1e-3 and want[0, 1, 1] > 1e-3


@pytest.mark.parametrize("ic", ["aic", "bic"])
def test_the_reads_are_at_the_selected_order(ic):
    """Under ``select`` the reads are of the order in force: ``VAR.fit`` at
    ``select_order``'s pick, on the rows that order's fit reads (the rows
    after the largest order's lags, as ``select_order`` fits every order
    on them), within the solve's bound."""
    k, pmax, h = 3, 5, 10
    df = _var(1000, k, 2, 6, exog=False)
    names = [f"y{i}" for i in range(k)]
    spec = rp.spec.vector_ar("v", targets=names, lags=pmax, select=ic, half_life=INF)
    bank = rp.ModelBank([spec])
    bank.fit(df)
    ys = df.select(names).to_numpy()
    order = int(VAR(ys).select_order(pmax).selected_orders[ic])
    assert bank.vector_ar(table="order").filter("in_force")["order"][0] == order
    res = VAR(ys[pmax - order :]).fit(order)
    b = _bound(_design(ys[pmax - order :], order, None, True))
    plain = bank.vector_ar(table="irf", periods=h)
    assert _rel(_cube(plain, "value", k), res.irf(h).irfs) <= (h + 1) * b
    fevd = bank.vector_ar(table="fevd", periods=h).sort("response", "horizon", "shock")
    got = fevd["share"].to_numpy().reshape(k, h, k)
    assert _rel(got, res.fevd(h).decomp) <= 2 * (h + 1) * b
