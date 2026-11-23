"""The cost of one STL or MSTL kernel row by the adjoint, against the port
on the identity and against one exact fit, at MSTL's hourly defaults.

Without the robust loop, STL and MSTL are linear in the window, so a
component at one position is one row ``e_p^T K`` of a ``W x W`` operator.
The port on the identity computes all of ``K`` (``W`` fits at once). The
adjoint runs ``e_p`` back through each step's transpose: one pass, the cost
of about one fit. Every step is a sparse matrix built from the port's own
loess weights (``LoessPlan._weights``), so the adjoint is checked against
the identity port's row at a small ``W`` before it is timed at the large one.

The port is the study's (branch ``vm/study-490``,
``scripts/study_490_489.py``), passed by path:

    git show origin/vm/study-490:scripts/study_490_489.py > /tmp/study.py
    uv run python scripts/stl_adjoint_490.py --study /tmp/study.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from typing import Any

import numpy as np
import scipy.sparse as sp


def _load(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("study_490_489", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plan_matrix(plan: Any, rows: int, cols: int, row_of: np.ndarray) -> sp.csr_matrix:
    """A loess plan's weights as a sparse map from its series to the rows
    ``row_of`` of the output."""
    w, ok = plan._weights(None)
    assert ok.all()
    r = np.repeat(row_of, plan.idx.shape[1])
    c = plan.idx.ravel()
    v = np.where(plan.mask, w, 0.0).ravel()
    keep = v != 0
    return sp.csr_matrix((v[keep], (r[keep], c[keep])), shape=(rows, cols))


def _ess_matrix(ess: Any) -> sp.csr_matrix:
    assert ess.nj == 1, "only jumps of 1 (the defaults) are built here"
    return _plan_matrix(ess.plan, ess.n, ess.n, np.arange(ess.pts.size))


def _moving_average(m: int, length: int) -> sp.csr_matrix:
    rows = m - length + 1
    r = np.repeat(np.arange(rows), length)
    c = (np.arange(rows)[:, None] + np.arange(length)[None, :]).ravel()
    return sp.csr_matrix((np.full(r.size, 1.0 / length), (r, c)), shape=(rows, m))


class StlSteps:
    """One STL port's linear steps: ``S = Sm (Y - T)``, ``T = Tr (Y - S)``,
    with ``Sm = (Csel - LP MA) Asub``."""

    def __init__(self, port: Any):
        n, p = port.n, port.period
        blocks = []
        for j, k, ess, left, right in port.sub:
            rows = j + p * np.arange(k + 2)  # the subseries' rows in C, ends included
            cols = j + p * np.arange(k)  # its rows in D
            mid = _ess_matrix(ess)
            lo = _plan_matrix(left, 1, k, np.array([0]))
            hi = _plan_matrix(right, 1, k, np.array([0]))
            local = sp.vstack([lo, mid, hi]).tocoo()
            blocks.append((rows[local.row], cols[local.col], local.data))
        r = np.concatenate([b[0] for b in blocks])
        c = np.concatenate([b[1] for b in blocks])
        v = np.concatenate([b[2] for b in blocks])
        self.asub = sp.csr_matrix((v, (r, c)), shape=(n + 2 * p, n))
        m = n + 2 * p
        ma1 = _moving_average(m, p)
        ma2 = _moving_average(m - p + 1, p)
        ma3 = _moving_average(m - 2 * p + 2, 3)
        self.ma = (ma3 @ ma2 @ ma1).tocsr()
        self.lp = _ess_matrix(port.lowpass)
        self.tr = _ess_matrix(port.trendess)
        self.csel = sp.csr_matrix(
            (np.ones(n), (np.arange(n), np.arange(p, p + n))), shape=(n, n + 2 * p)
        )
        self.asub_t = self.asub.T.tocsr()
        self.ma_t = self.ma.T.tocsr()
        self.lp_t = self.lp.T.tocsr()
        self.tr_t = self.tr.T.tocsr()
        self.csel_t = self.csel.T.tocsr()

    def sm(self, d: np.ndarray) -> np.ndarray:
        c = self.asub @ d
        return self.csel @ c - self.lp @ (self.ma @ c)

    def sm_t(self, g: np.ndarray) -> np.ndarray:
        return self.asub_t @ (self.csel_t @ g - self.ma_t @ (self.lp_t @ g))

    def fit(self, y: np.ndarray, inner: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.zeros_like(y)
        s = np.zeros_like(y)
        for _ in range(inner):
            s = self.sm(y - t)
            t = self.tr @ (y - s)
        return s, t

    def adjoint(self, g_s: np.ndarray, g_t: np.ndarray, inner: int) -> np.ndarray:
        """``S^T g_s + T^T g_t`` for the fit's outputs ``S`` and ``T``: the
        loop of ``fit`` run backwards, each step transposed."""
        g_y = np.zeros_like(g_s)
        for _ in range(inner):
            # t_k = Tr (y - s_k): the output t's gradient flows to y and to s_k
            u = self.tr_t @ g_t
            g_y += u
            g_s = g_s - u
            # s_k = Sm (y - t_{k-1}): to y, and to the previous t
            v = self.sm_t(g_s)
            g_y += v
            g_t = -v
            g_s = np.zeros_like(g_s)
        return g_y  # t_0 = 0 takes no gradient


def mstl_fit(steps: list[StlSteps], y: np.ndarray, iterate: int, inner: int):
    seasonal = [np.zeros_like(y) for _ in steps]
    deseas = y.copy()
    trend = None
    for _ in range(iterate if len(steps) > 1 else 1):
        for i, st in enumerate(steps):
            deseas = deseas + seasonal[i]
            s, trend = st.fit(deseas, inner)
            seasonal[i] = s
            deseas = deseas - s
    return seasonal, trend


def mstl_adjoint(
    steps: list[StlSteps], g_seas: list[np.ndarray], g_trend: np.ndarray, iterate: int, inner: int
) -> np.ndarray:
    """``sum_i S_i^T g_i + T^T g_T`` for MSTL's outputs, reversing
    ``mstl_fit``'s chain of updates."""
    n = g_trend.size
    passes = iterate if len(steps) > 1 else 1
    calls = [(it, i) for it in range(passes) for i in range(len(steps))]
    g_s = [g.copy() for g in g_seas]  # the gradient on each seasonal's latest value
    g_d = np.zeros(n)  # on deseas's latest value
    g_t_last = g_trend.copy()  # only the last call's trend is an output
    for idx, (_it, i) in enumerate(reversed(calls)):
        # deseas_out = deseas_in' - s_new; seasonal[i] = s_new; deseas_in' = deseas_in + s_old
        g_snew = g_s[i] - g_d
        g_t = g_t_last if idx == 0 else np.zeros(n)
        g_in = steps[i].adjoint(g_snew, g_t, inner)  # onto deseas_in'
        g_d = g_d + g_in
        g_s[i] = g_d.copy()  # s_old entered deseas_in' with coefficient 1
    return g_d  # deseas starts as y


def _time(fn, budget: float = 3.0) -> float:
    n, t0 = 0, time.perf_counter()
    while True:
        fn()
        n += 1
        el = time.perf_counter() - t0
        if el > budget or n >= 50:
            return el / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--check-w", type=int, default=401)
    ap.add_argument("--w", type=int, default=2353)
    ap.add_argument("--identity-columns", type=int, default=64)
    args = ap.parse_args()
    st = _load(args.study)
    periods, windows, iterate, inner = [24, 168], [11, 15], 2, 5
    out: dict[str, Any] = {
        "periods": periods,
        "windows": windows,
        "iterate": iterate,
        "inner": inner,
    }

    # ---- the adjoint against the port on the identity, at a small W
    w = args.check_w
    ports = [st.STLPort(w, p, seasonal=s) for p, s in zip(periods, windows, strict=True)]
    steps = [StlSteps(p) for p in ports]
    ref = st.mstl_port(np.eye(w), periods, windows, iterate=iterate, ports=ports)
    pos = w // 2
    gap = 0.0
    for comp in range(3):  # seasonal 24, seasonal 168, trend
        g_seas = [np.zeros(w), np.zeros(w)]
        g_trend = np.zeros(w)
        if comp < 2:
            g_seas[comp][pos] = 1.0
            row_ref = ref["seasonal"][comp][pos]
        else:
            g_trend[pos] = 1.0
            row_ref = ref["trend"][pos]
        row = mstl_adjoint(steps, g_seas, g_trend, iterate, inner)
        gap = max(gap, float(np.max(np.abs(row - row_ref))))
    y = np.random.default_rng(490).standard_normal(w)
    fwd = mstl_fit(steps, y, iterate, inner)
    port_fit = st.mstl_port(y, periods, windows, iterate=iterate, ports=ports)
    out["check"] = {
        "W": w,
        "position": pos,
        "adjoint_vs_identity_port_max_abs": gap,
        "sparse_forward_vs_port_max_abs": float(np.max(np.abs(fwd[1] - port_fit["trend"]))),
    }

    # ---- the costs at W
    w = args.w
    t0 = time.perf_counter()
    ports = [st.STLPort(w, p, seasonal=s) for p, s in zip(periods, windows, strict=True)]
    t_ports = time.perf_counter() - t0
    t0 = time.perf_counter()
    steps = [StlSteps(p) for p in ports]
    t_steps = time.perf_counter() - t0
    pos = w // 2
    y = np.random.default_rng(491).standard_normal(w)

    def adjoint_rows() -> None:  # the three kernel rows an emission dots
        for comp in range(3):
            g_seas = [np.zeros(w), np.zeros(w)]
            g_trend = np.zeros(w)
            (g_seas[comp] if comp < 2 else g_trend)[pos] = 1.0
            mstl_adjoint(steps, g_seas, g_trend, iterate, inner)

    def adjoint_one() -> None:
        g_seas = [np.zeros(w), np.zeros(w)]
        g_seas[0][pos] = 1.0
        mstl_adjoint(steps, g_seas, np.zeros(w), iterate, inner)

    t_adj1 = _time(adjoint_one)
    t_adj3 = _time(adjoint_rows)
    t_fit = _time(lambda: st.mstl_port(y, periods, windows, iterate=iterate, ports=ports))
    t_sparse_fit = _time(lambda: mstl_fit(steps, y, iterate, inner))
    k = args.identity_columns
    block = np.eye(w)[:, :k]
    t_block = _time(
        lambda: st.mstl_port(block, periods, windows, iterate=iterate, ports=ports), budget=10.0
    )
    out["at_W"] = {
        "W": w,
        "build_port_s": t_ports,
        "build_sparse_steps_s": t_steps,
        "adjoint_one_row_ms": t_adj1 * 1e3,
        "adjoint_three_rows_ms": t_adj3 * 1e3,
        "port_one_fit_ms": t_fit * 1e3,
        "sparse_one_fit_ms": t_sparse_fit * 1e3,
        "port_identity_block_columns": k,
        "port_identity_block_s": t_block,
        "port_identity_whole_s_scaled": t_block * w / k,
        "nnz_per_stl": [int(s.asub.nnz + s.ma.nnz + s.lp.nnz + s.tr.nnz) for s in steps],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
