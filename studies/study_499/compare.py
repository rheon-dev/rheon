"""Each estimator against hfhd on the same ticks: the checks behind the
write-up's tables. Every hfhd call states its settings."""

from __future__ import annotations

import math
import warnings

import numba
import numpy as np
from hfhd import hf

import data as D
import estimators as E

warnings.filterwarnings("ignore", module="numba")
warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)


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
                kname = {1: "K=None (int(N_p^(2/3)))", 2: "K = int(N_p^(2/3))", 5: "K = int(N_p^(2/3)) // 2"}[J]
                put(("tsrc", f"J={J}, {kname}", label),
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


# --------------------------------------------------------------------------
# where the paper and hfhd part
# --------------------------------------------------------------------------


def mrc_flags(seed: int = 0) -> list[dict]:
    """The bias-corrected MRC at hfhd's defaults (theta 0.4), from hfhd's
    convention to the paper's one flag at a time; each row the change it
    alone makes, relative to the estimate's largest entry."""
    day = D.simulate(seed)
    out = []
    for label, P, _ in grids(day)[:2]:
        n = P.shape[0] - 1
        k = E.mrc_k(n, 0.4, True)
        base = E.mrc(P, k, True, "hfhd", False, False, theta=0.4)
        flips = [
            ("windows from the first return (n - k + 2 of them)", dict(windows="paper")),
            ("psi_1 summed to k, not k - 1", dict(psi1_upto_k=True)),
            ("rescaled by 1 / (1 - psi_1 / (2 psi_2 k^2))", dict(rescale=True)),
            ("theta in the bias term = k / sqrt(n), not the input 0.4", dict(theta=None)),
        ]
        cfg = dict(windows="hfhd", psi1_upto_k=False, rescale=False, theta=0.4)
        for name, f in flips:
            c = {**cfg, **f}
            out.append({"grid": label, "k": k, "flag": name,
                        "gap": rel(E.mrc(P, k, True, **c), base)})
        out.append({"grid": label, "k": k, "flag": "all four (the paper's form)",
                    "gap": rel(E.mrc(P, k, True), base)})
    return out


def hy_preavg_gaps(seeds) -> list[dict]:
    """The pre-averaged HY covariance, paper's index convention against
    hfhd's, beside the estimator's own spread over the seeds and the truth."""
    out = []
    for th in (0.4, 0.6):
        rows = []
        for seed in seeds:
            day = D.simulate(seed)
            k = E.hy_k(len(day.a_t), len(day.b_t), th)
            h = E.hy_preavg(day.a_t, day.a_p, day.b_t, day.b_p, k, "hfhd")
            p = E.hy_preavg(day.a_t, day.a_p, day.b_t, day.b_p, k, "paper")
            pe = E.hy_preavg(day.a_t, day.a_p, day.b_t, day.b_p, k, "paper", psi="exact")
            rows.append((h / day.iv[0, 1], p / day.iv[0, 1], pe / day.iv[0, 1], k))
        r = np.array(rows)
        out.append({"theta": th, "k": f"{int(r[:, 3].min())}-{int(r[:, 3].max())}",
                    "hfhd_mean": r[:, 0].mean(), "paper_mean": r[:, 1].mean(),
                    "paper_exact_psi_mean": r[:, 2].mean(),
                    "gap_mean": (r[:, 1] - r[:, 0]).mean(), "gap_sd": (r[:, 1] - r[:, 0]).std(ddof=1),
                    "est_sd": r[:, 0].std(ddof=1)})
    return out


def bandwidths(seed: int = 0) -> list[dict]:
    day = D.simulate(seed)
    out = []
    for label, P, _ in grids(day)[:2]:
        n = P.shape[0] - 1
        om, iv = E.noise_and_iv(P)
        for kern in ("parzen", "quadratic_spectral"):
            Hh = hf.get_bandwidth(n, float(iv[0]), float(om[0]), kern)
            Hx = E.bandwidth(n, float(iv[0]), float(om[0]), kern)
            Hc = int(math.ceil(Hx))
            a = E.krvm(P, Hh, kern, 1, "bnhls")
            b = E.krvm(P, Hc, kern, 1, "bnhls")
            out.append({"grid": label, "kernel": kern, "n": n, "xi2": float(om[0] / iv[0]),
                        "c_hfhd": 3.51 if kern == "parzen" else 0.46, "c_exact": E.c_star(kern),
                        "H_hfhd": Hh, "H_exact": Hx, "H_ceil": Hc, "estimate_moves": rel(a, b)})
    return out


def qs_truncation(seed: int = 0) -> list[dict]:
    """The quadratic spectral kernel weighs every lag; truncated at ``L``
    lags, how far from the full sum, and what state the one-pass form keeps."""
    day = D.simulate(seed)
    out = []
    for label, P, _ in grids(day)[:2]:
        n = P.shape[0] - 1
        d = P.shape[1]
        om, iv = E.noise_and_iv(P)
        H = max(hf.get_bandwidth(n, float(iv[0]), float(om[0]), "quadratic_spectral"), 1)
        for form in ("bnhls", "flat_top"):
            full = E.krvm(P, H, "quadratic_spectral", 1, form)
            for mult in (1, 2, 5, 10, 20, 50, 100, 400):
                L = min(mult * H, n - 1)
                cut = E.krvm(P, H, "quadratic_spectral", 1, form, lags=L)
                out.append({"grid": label, "form": form, "H": H, "L": L, "L/H": mult,
                            "gap": rel(cut, full), "state_bytes": 8 * (L * d + d + d * d + 1)})
    return out


# --------------------------------------------------------------------------
# against the truth
# --------------------------------------------------------------------------


def accuracy(seeds) -> list[dict]:
    """Each estimator's error against the simulated day's integrated
    (co)variance over the seeds: mean of estimate / truth - 1 and its root
    mean square, for asset A's variance and the covariance."""
    acc: dict[tuple, list] = {}

    def add(key, est, iv):
        acc.setdefault(key, []).append((est[0, 0] / iv[0, 0] - 1 if est.shape[0] else np.nan,
                                        est[0, 1] / iv[0, 1] - 1 if est.shape[1] > 1 else np.nan))

    for seed in seeds:
        day = D.simulate(seed)
        for label, P, _ in grids(day)[:2]:
            n = P.shape[0] - 1
            om, iv = E.noise_and_iv(P)
            K = int((n + 1) ** (2 / 3))
            Md = int(np.ceil((n + 1) ** 0.5))
            add((label, "realised covariance (msrc M=1)"), E.msrc(P, 1, 0), day.iv)
            add((label, "tsrc, K = n^(2/3), J = 1"), E.tsrc(P, K, 1, "hfhd"), day.iv)
            add((label, "tsrc, the same, ZMA's small-sample adjustment"), E.tsrc(P, K, 1, "paper", True), day.iv)
            add((label, "msrc, M = N = sqrt(n)"), E.msrc(P, Md, Md), day.iv)
            k = E.mrc_k(n, 0.4, True)
            add((label, "mrc, theta 0.4, bias-corrected (hfhd)"), E.mrc(P, k, True, "hfhd", False, False, theta=0.4), day.iv)
            add((label, "mrc, theta 0.4, bias-corrected (paper)"), E.mrc(P, k, True), day.iv)
            k = E.mrc_k(n, 1.0, False)
            add((label, "mrc, theta 1.0, no bias term (rcov's default)"), E.mrc(P, k, False), day.iv)
            for kern in ("parzen", "quadratic_spectral"):
                Hh = max(hf.get_bandwidth(n, float(iv[0]), float(om[0]), kern), 1)
                add((label, f"krvm {kern}, flat-top (hfhd), H = get_bandwidth"), E.krvm(P, Hh, kern, 1, "flat_top"), day.iv)
                Hx = math.ceil(E.bandwidth(n, float(iv[0]), float(om[0]), kern))
                add((label, f"krvm {kern}, BNHLS, jitter 2, H = ceil(exact)"), E.krvm(P, Hx, kern, 2, "bnhls"), day.iv)
        hyv = E.hy(day.a_t, day.a_p, day.b_t, day.b_p)
        add(("ticks", "hayashi_yoshida, k = 1"), np.array([[np.nan, hyv], [hyv, np.nan]]), day.iv)
        k = E.hy_k(len(day.a_t), len(day.b_t), 0.4)
        for conv in ("hfhd", "paper"):
            v = E.hy_preavg(day.a_t, day.a_p, day.b_t, day.b_p, k, conv)
            add(("ticks", f"hayashi_yoshida, theta 0.4 ({conv})"), np.array([[np.nan, v], [v, np.nan]]), day.iv)
    out = []
    for (g, name), v in acc.items():
        a = np.array(v)
        out.append({"grid": g, "estimator": name,
                    "var_bias": np.nanmean(a[:, 0]), "var_rmse": float(np.sqrt(np.nanmean(a[:, 0] ** 2))),
                    "cov_bias": np.nanmean(a[:, 1]), "cov_rmse": float(np.sqrt(np.nanmean(a[:, 1] ** 2)))})
    return out


# --------------------------------------------------------------------------
# state and cost
# --------------------------------------------------------------------------


def wide(seed: int, d: int, n: int = 23_400) -> np.ndarray:
    """A synchronous grid of ``d`` noisy log-prices (equicorrelated 0.5): for
    timing only, at the same noise ratio as the two-asset day."""
    rng = np.random.default_rng(seed)
    c = 0.5 * np.ones((d, d)) + 0.5 * np.eye(d)
    z = rng.standard_normal((n, d)) @ np.linalg.cholesky(c).T * (D.SIGMA / math.sqrt(n))
    x = np.vstack([np.zeros((1, d)), np.cumsum(z, axis=0)]) + math.log(100.0)
    return x + D.OMEGA * rng.standard_normal(x.shape)


def configs(n: int, d: int, iv=None, om=None):
    """Each one-pass form at hfhd's default settings for ``n`` returns:
    (name, settings, formula, bytes by formula, new, feed, close)."""
    K = int((n + 1) ** (2 / 3))
    Md = int(np.ceil((n + 1) ** 0.5))
    k1 = E.mrc_k(n, 0.4, True)
    k2 = E.mrc_k(n, 1.0, False)
    Hp = 55 if iv is None else max(hf.get_bandwidth(n, iv, om, "parzen"), 1)
    Hq = 7 if iv is None else max(hf.get_bandwidth(n, iv, om, "quadratic_spectral"), 1)
    Hmax = int(math.ceil(E.c_star("parzen") * n**0.6))  # the ring at xi = 1, as rcov sizes it
    wp = S.krvm_weights(Hp, "parzen", "flat_top")
    wm = S.krvm_weights(Hmax, "parzen", "bnhls")
    wq = S.krvm_weights(Hq, "quadratic_spectral", "bnhls", lags=10 * Hq)
    L = lambda w: len(w) - 1  # noqa: E731
    return [
        ("tsrc", f"K = {K}, J = 1", "8 (K d + 2 d^2 + 1)", 8 * (K * d + 2 * d * d + 1),
         lambda: S.tsrc_new(d, K), lambda st, x: S.tsrc_feed(st, x, K), lambda st: S.tsrc_close(st, K)),
        ("msrc", f"M = N = {Md}", "8 ((N + M) d + 3 d^2 + 1)", 8 * (2 * Md * d + 3 * d * d + 1),
         lambda: S.msrc_new(d, Md, Md), lambda st, x: S.msrc_feed(st, x, Md, Md), lambda st: S.msrc_close(st, Md, Md)),
        ("mrc", f"theta 0.4, bias-corrected, k = {k1}", "8 ((k - 1) d + d + 2 d^2 + 2)", 8 * ((k1 - 1) * d + d + 2 * d * d + 2),
         lambda: S.mrc_new(d, k1), lambda st, x: S.mrc_feed(st, x, k1), lambda st: S.mrc_close(st, k1, True)),
        ("mrc", f"theta 1.0, no bias term, k = {k2}", "8 ((k - 1) d + d + 2 d^2 + 2)", 8 * ((k2 - 1) * d + d + 2 * d * d + 2),
         lambda: S.mrc_new(d, k2), lambda st, x: S.mrc_feed(st, x, k2), lambda st: S.mrc_close(st, k2, False)),
        ("krvm", f"Parzen, H = {Hp} fixed, one accumulator", "8 (H d + d + d^2 + 1)", 8 * (L(wp) * d + d + d * d + 1),
         lambda: S.krvm_new(d, wp), lambda st, x: S.krvm_feed(st, x, wp), lambda st: S.krvm_close(st, wp)),
        ("krvm", f"Parzen, H chosen at close, ring {Hmax}", "8 (H_max d + d + (H_max + 1) d^2 + 1)",
         8 * (L(wm) * d + d + (L(wm) + 1) * d * d + 1),
         lambda: S.krvm_new(d, wm, True), lambda st, x: S.krvm_feed(st, x, wm, True), lambda st: S.krvm_close(st, wm, True)),
        ("krvm", f"QS, H = {Hq}, cut at L = 10 H = {L(wq)}", "8 (L d + d + d^2 + 1)", 8 * (L(wq) * d + d + d * d + 1),
         lambda: S.krvm_new(d, wq), lambda st, x: S.krvm_feed(st, x, wq), lambda st: S.krvm_close(st, wq)),
    ]


def state_and_cost(d_list=(2, 10), repeat: int = 3) -> list[dict]:
    """For each one-pass form: the state by formula and by counting the
    arrays after a day's rows, and the cost per tick of hfhd's call, the
    numpy batch definition and the compiled one-pass form (best of
    ``repeat``, after a warm-up call that compiles)."""
    out = []

    def best(f):
        f()
        ts = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    for d in d_list:
        if d == 2:
            day = D.simulate(0)
            P = day.sync_p
        else:
            P = wide(1, d)
        n = P.shape[0] - 1
        ser = series_of(P)
        om, iv = E.noise_and_iv(P)
        K = int((n + 1) ** (2 / 3))
        Md = int(np.ceil((n + 1) ** 0.5))
        for name, setting, formula, fb, new, feed, close in configs(n, d, float(iv[0]), float(om[0])):
            st = new()
            feed(st, P)
            close(st)
            t_stream = best(lambda: stream(new, feed, close, P, 1))
            if name == "tsrc":
                t_hf = best(lambda: hf.tsrc(ser))
                t_np = best(lambda: E.tsrc(P, K, 1, "hfhd"))
            elif name == "msrc":
                t_hf = best(lambda: hf.msrc(ser, pairwise=False))
                t_np = best(lambda: E.msrc(P, Md, Md))
            elif name == "mrc":
                th, bc = (0.4, True) if "bias-corrected" in setting else (1.0, False)
                kk = E.mrc_k(n, th, bc)
                t_hf = best(lambda: hf.mrc(ser, theta=th, bias_correction=bc, pairwise=False))
                t_np = best(lambda: E.mrc(P, kk, bc, "hfhd", False, False, theta=th))
            elif "QS" in setting:
                Hq = max(hf.get_bandwidth(n, float(iv[0]), float(om[0]), "quadratic_spectral"), 1)
                t_hf = best(lambda: hf.krvm(ser, Hq, pairwise=False, kernel=hf.quadratic_spectral_kernel)) if d == 2 else None
                t_np = best(lambda: E.krvm(P, Hq, "quadratic_spectral", 1, "bnhls")) if d == 2 else None
            else:
                Hp = max(hf.get_bandwidth(n, float(iv[0]), float(om[0]), "parzen"), 1)
                t_hf = best(lambda: hf.krvm(ser, Hp, pairwise=False, kernel=hf.parzen_kernel))
                t_np = best(lambda: E.krvm(P, Hp, "parzen", 1, "flat_top"))
            out.append({"estimator": name, "setting": setting, "d": d, "n": n, "formula": formula,
                        "bytes_formula": fb, "bytes_counted": S.nbytes(st),
                        "ns_hfhd": None if t_hf is None else 1e9 * t_hf / n,
                        "ns_numpy": None if t_np is None else 1e9 * t_np / n,
                        "ns_one_pass": 1e9 * t_stream / n})
    # the tick-level ones: refresh time and Hayashi-Yoshida on the asynchronous pair
    day = D.simulate(0)
    sa, sb = day.series("async")
    t, a, p = S.long_frame([day.a_t, day.b_t], [day.a_p, day.b_p])
    nt = len(t)
    _, _, rst = S.rt_run(t, a, p, 2)
    hst = S.hy_run(t, a, p)
    out.append({"estimator": "refresh_time", "setting": "2 assets, a long frame", "d": 2, "n": nt,
                "formula": "8 (3 + k) + k", "bytes_formula": 8 * (3 + 2) + 2, "bytes_counted": S.nbytes(rst),
                "ns_hfhd": 1e9 * best(lambda: hf.refresh_time([sa, sb])) / nt,
                "ns_numpy": 1e9 * best(lambda: E.refresh_time([day.a_t, day.b_t], [day.a_p, day.b_p])) / nt,
                "ns_one_pass": 1e9 * best(lambda: S.rt_run(t, a, p, 2)) / nt})
    out.append({"estimator": "hayashi_yoshida", "setting": "one pair, k = 1", "d": 2, "n": nt,
                "formula": "8 (13 + 2 + 3) + 2", "bytes_formula": 8 * 18 + 2, "bytes_counted": S.nbytes(hst),
                "ns_hfhd": 1e9 * best(lambda: hf.hayashi_yoshida([sa, sb])) / nt,
                "ns_numpy": 1e9 * best(lambda: E.hy(day.a_t, day.a_p, day.b_t, day.b_p)) / nt,
                "ns_one_pass": 1e9 * best(lambda: S.hy_run(t, a, p)) / nt})
    kk = E.hy_k(len(day.a_t), len(day.b_t), 0.4)
    out.append({"estimator": "hayashi_yoshida", "setting": f"pre-averaged, theta 0.4 (k = {kk})", "d": 2, "n": nt,
                "formula": "not one pass with a bound (see text)", "bytes_formula": None, "bytes_counted": None,
                "ns_hfhd": 1e9 * best(lambda: hf.hayashi_yoshida([sa, sb], theta=0.4)) / nt,
                "ns_numpy": 1e9 * best(lambda: E.hy_preavg(day.a_t, day.a_p, day.b_t, day.b_p, kk, "hfhd")) / nt,
                "ns_one_pass": None})
    return out
