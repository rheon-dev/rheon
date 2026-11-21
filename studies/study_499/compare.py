"""Each estimator against hfhd on the same ticks: the checks behind the
write-up's tables. Every hfhd call states its settings."""

from __future__ import annotations

import warnings

import numpy as np
from hfhd import hf

import data as D
import estimators as E

warnings.filterwarnings("ignore", module="numba")


def _indices_and_values(tick_series_list):
    """hfhd 0.1.4's ``_get_indeces_and_values`` with one change: it pads the
    uint64 stamp matrix with ``np.nan``, which numpy 2 refuses ("cannot
    convert float NaN to integer"), so every pairwise path and
    ``hayashi_yoshida`` raise. This pads with 0. hfhd never reads the
    padding: each row is cut at the count of non-NaN *values*, which keep
    their NaN padding as before."""
    n_max = np.max([len(x) for x in tick_series_list])
    idx = np.zeros((len(tick_series_list), n_max), dtype="uint64")
    val = np.full((len(tick_series_list), n_max), np.nan)
    for i, x in enumerate(tick_series_list):
        t = np.array(x.dropna().index, dtype="uint64")
        v = np.array(x.dropna().to_numpy(), dtype="float64")
        idx[i, : t.shape[0]] = t
        val[i, : t.shape[0]] = v
    return idx, val


hf._get_indeces_and_values = _indices_and_values


def rel(a, b) -> float:
    """Largest absolute difference over the largest absolute entry of ``b``."""
    a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
    return float(np.abs(a - b).max() / np.abs(b).max())


import time  # noqa: E402

import numba  # noqa: E402
import pandas as pd  # noqa: E402

import streaming as S  # noqa: E402


def hfhd_kernel(name: str, H: float):
    """BNHLS 2011's weight ``k(h / (H + 1))``, compiled for hfhd's ``krvm``,
    which weighs lag ``h`` by ``kernel((h - 1) / H_hfhd)``: called with
    ``H_hfhd = 1`` the function receives ``h - 1``."""
    scale = float(H + 1)
    if name == "parzen":

        @numba.njit
        def kern(u):
            x = abs((u + 1.0) / scale)
            if x <= 0.5:
                return 1.0 - 6.0 * x**2 + 6.0 * x**3
            if x <= 1.0:
                return 2.0 * (1.0 - x) ** 3
            return 0.0

        return kern

    @numba.njit
    def kern_qs(u):
        x = abs((u + 1.0) / scale)
        return 3.0 / x**2 * (np.sin(x) / x - np.cos(x))

    return kern_qs


HF_KERNEL = {"parzen": hf.parzen_kernel, "quadratic_spectral": hf.quadratic_spectral_kernel}


def series_of(p: np.ndarray) -> list[pd.Series]:
    idx = D.to_index(D.OPEN_NS + np.arange(p.shape[0], dtype=np.int64) * 10**9)
    return [pd.Series(p[:, j], index=idx) for j in range(p.shape[1])]


def stream(new, feed, close, p, parts):
    st = new()
    for s in S.chunks(len(p), parts):
        feed(st, p[s])
    return close(st), st


def stream_gap(new, feed, close, p, want):
    """The one-pass form against ``want``, and whether 1 and 997 chunks give
    the same bits."""
    one, st = stream(new, feed, close, p, 1)
    many, _ = stream(new, feed, close, p, 997)
    return rel(one, want), bool(np.array_equal(one, many)), S.nbytes(st)


def grids(day):
    """(label, price matrix, hfhd input) for the synchronous grid, the
    refresh-time grid of the asynchronous pair, and asset A alone on its
    irregular ticks (hfhd's refresh-time estimators get it twice)."""
    rt = E.refresh_time([day.a_t, day.b_t], [day.a_p, day.b_p])[1]
    a = day.a_p[:, None]
    sa = day.series("irregular")[0]
    return [
        ("sync", day.sync_p, day.series("sync")),
        ("async", rt, day.series("async")),
        ("irregular", a, [sa, sa.copy()]),
    ]


def agreement(seeds) -> list[dict]:
    """Every estimator, every setting, every grid: the hfhd-convention batch
    and one-pass forms against hfhd, and the paper's form against hfhd."""
    rows: dict[tuple, dict] = {}

    def put(key, hf_gap, st_gap=None, inv=None, paper_gap=None, nb=None):
        r = rows.setdefault(key, {"hf": 0.0, "st": None, "inv": True, "paper": None, "bytes": nb})
        r["hf"] = max(r["hf"], hf_gap)
        if st_gap is not None:
            r["st"] = max(r["st"] or 0.0, st_gap)
            r["inv"] = r["inv"] and inv
        if paper_gap is not None:
            r["paper"] = max(r["paper"] or 0.0, paper_gap)

    for seed in seeds:
        day = D.simulate(seed)
        sa, sb = day.series("async")
        # refresh time
        ht = hf.refresh_time([sa, sb])
        st_, sv = E.refresh_time([day.a_t, day.b_t], [day.a_p, day.b_p])
        t, a, p = S.long_frame([day.a_t, day.b_t], [day.a_p, day.b_p])
        ot, ov, rst = S.rt_run(t, a, p, 2, 1)
        ot2, ov2, _ = S.rt_run(t, a, p, 2, 997)
        same = np.array_equal(np.array(ht.index, dtype=np.int64), st_) and np.array_equal(ht.to_numpy(), sv)
        put(("refresh_time", "-", "async"), 0.0 if same else 1.0,
            0.0 if (np.array_equal(ot, st_) and np.array_equal(ov, sv)) else 1.0,
            np.array_equal(ot, ot2) and np.array_equal(ov, ov2), nb=S.nbytes(rst))
        # Hayashi-Yoshida
        h = hf.hayashi_yoshida([sa, sb])
        mine = E.hy(day.a_t, day.a_p, day.b_t, day.b_p)
        hst = S.hy_run(t, a, p, 1)
        hst2 = S.hy_run(t, a, p, 997)
        put(("hayashi_yoshida", "theta=None (k=1)", "async"), rel(mine, h[0, 1]),
            rel(hst["hy"][0], h[0, 1]), np.array_equal(hst["hy"], hst2["hy"]), nb=S.nbytes(hst))
        put(("hayashi_yoshida", "diagonal", "async"),
            max(rel(hst["hy"][1], h[0, 0]), rel(hst["hy"][2], h[1, 1])))
        for th in (0.4, 0.6):
            k = E.hy_k(len(day.a_t), len(day.b_t), th)
            h = hf.hayashi_yoshida([sa, sb], theta=th)
            put(("hayashi_yoshida", f"theta={th} (k={k})", "async"),
                rel(E.hy_preavg(day.a_t, day.a_p, day.b_t, day.b_p, k, "hfhd"), h[0, 1]),
                paper_gap=rel(E.hy_preavg(day.a_t, day.a_p, day.b_t, day.b_p, k, "paper"), h[0, 1]))
        for label, P, ser in grids(day):
            d = P.shape[1]
            n = P.shape[0] - 1
            # two scales
            K = int((n + 1) ** (2 / 3))
            for J, KK in ((1, K), (2, K), (5, max(K // 2, 6))):
                h = hf.tsrc(ser, J=J, K=KK if (J, KK) != (1, K) else None)
                h = h[:d, :d]
                g = stream_gap(lambda: S.tsrc_new(d, KK, J), lambda st, x: S.tsrc_feed(st, x, KK, J),
                               lambda st: S.tsrc_close(st, KK, J), P, h)
                put(("tsrc", f"J={J}, K={'default' if (J, KK) == (1, K) else KK}", label),
                    rel(E.tsrc(P, KK, J, "hfhd"), h), g[0], g[1], rel(E.tsrc(P, KK, J, "paper"), h), g[2])
            # multi scale
            Md = int(np.ceil((n + 1) ** 0.5))
            for M, N, name in ((Md, Md, "M=N=None (default)"), (10, 0, "M=10, N=0"), (1, 0, "M=1")):
                kw = {} if name.endswith("(default)") else {"M": M, "N": N}
                h = hf.msrc(ser, pairwise=False, **kw)[:d, :d]
                g = stream_gap(lambda: S.msrc_new(d, M, N), lambda st, x: S.msrc_feed(st, x, M, N),
                               lambda st: S.msrc_close(st, M, N), P, h)
                put(("msrc", name + ", pairwise=False", label), rel(E.msrc(P, M, N), h), g[0], g[1],
                    rel(E.msrc(P, M, N, n_conv="returns"), h), g[2])
            # pre-averaging
            ser1 = ser[:1] if label == "irregular" else ser
            for th, bc in ((0.4, True), (0.6, True), (1.0, False)):
                k = E.mrc_k(n, th, bc)
                h = hf.mrc(ser1, theta=th, bias_correction=bc, pairwise=False)
                g = stream_gap(lambda: S.mrc_new(d, k), lambda st, x: S.mrc_feed(st, x, k, "hfhd"),
                               lambda st: S.mrc_close(st, k, bc, False, False, th), P, h)
                put(("mrc", f"theta={th}, bias_correction={bc}, pairwise=False", label),
                    rel(E.mrc(P, k, bc, "hfhd", False, False, theta=th), h), g[0], g[1],
                    rel(E.mrc(P, k, bc), h), g[2])
            # realised kernels
            om, iv = E.noise_and_iv(P)
            for kern in ("parzen", "quadratic_spectral"):
                H = hf.get_bandwidth(n, float(iv[0]), float(om[0]), kern)
                H = max(H, 1)
                h = hf.krvm(ser1, H, pairwise=False, kernel=HF_KERNEL[kern])
                lags = None if kern == "parzen" else n - 1
                w = S.krvm_weights(H, kern, "flat_top", lags)
                g = stream_gap(lambda: S.krvm_new(d, w), lambda st, x: S.krvm_feed(st, x, w),
                               lambda st: S.krvm_close(st, w), P, h) if kern == "parzen" else (None, None, None)
                put(("krvm", f"{kern}, flat-top k((h-1)/H), H=get_bandwidth", label),
                    rel(E.krvm(P, H, kern, 1, "flat_top"), h), g[0], g[1],
                    rel(E.krvm(P, H, kern, 1, "bnhls"), h), g[2])
                for m in (1, 2):
                    J_ = E.jitter(P, m)
                    h = hf.krvm(series_of(J_)[: (1 if label == "irregular" else d)], 1, pairwise=False,
                                kernel=hfhd_kernel(kern, H))
                    put(("krvm", f"{kern}, BNHLS k(h/(H+1)), jitter m={m} (kernel passed)", label),
                        rel(E.krvm(P, H, kern, m, "bnhls"), h))
        # pairwise = True on the asynchronous pair: each variance on its own ticks
        ser = [sa, sb]
        own = [day.a_p[:, None], day.b_p[:, None]]
        rt = E.refresh_time([day.a_t, day.b_t], [day.a_p, day.b_p])[1]
        Mm = lambda q: int(np.ceil(q.shape[0] ** 0.5))  # noqa: E731
        h = hf.msrc(ser, pairwise=True)
        want = np.array([[E.msrc(own[0], Mm(own[0]), Mm(own[0]))[0, 0], E.msrc(rt, Mm(rt), Mm(rt))[0, 1]],
                         [0.0, E.msrc(own[1], Mm(own[1]), Mm(own[1]))[0, 0]]])
        want[1, 0] = want[0, 1]
        put(("msrc", "M=N=None, pairwise=True", "async"), rel(want, h))
        h = hf.mrc(ser, theta=0.4, pairwise=True)

        def m1(q):
            nn = q.shape[0] - 1
            kk = E.mrc_k(nn, 0.4, True)
            return E.mrc(q, kk, True, "hfhd", False, False, theta=0.4)

        want = np.array([[m1(own[0])[0, 0], m1(rt)[0, 1]], [m1(rt)[0, 1], m1(own[1])[0, 0]]])
        put(("mrc", "theta=0.4, bias_correction=True, pairwise=True", "async"), rel(want, h))
    return [
        {"estimator": k[0], "setting": k[1], "grid": k[2], **v} for k, v in rows.items()
    ]
