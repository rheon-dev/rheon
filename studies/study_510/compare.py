"""Study 510's checks against scikit-learn, scipy and longhands; each returns its worst gap.

``python scripts/study_510.py`` prints the tables the record quotes;
``test_checks.py`` asserts the same checks. Gaps are absolute (every quantity
here is a number of order one or smaller) unless a name says ``rel``.
"""

from __future__ import annotations

import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import ndtr
from sklearn.metrics import mutual_info_score

sys.path.insert(0, str(Path(__file__).parent))
from dependence import (  # noqa: E402
    BinnedMI,
    RankWindow,
    chatterjee_xi,
    edges_from,
    mi_from_tables,
    run_ranks,
)

from data import gaussian, stream  # noqa: E402


def labels(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The oracle's bin labels: numpy's ``digitize`` (``edges[i-1] <= v < edges[i]``)."""
    return np.digitize(v, edges)


def window_rows(i: int, window: int | None) -> slice:
    """The rows the state holds before row ``i`` is learned."""
    return slice(0 if window is None else max(0, i - window), i)


def pair_sample(x, y, w, i, window, j, t):
    sl = window_rows(i, window)
    xs, ys, ws = x[sl, j], y[sl, t], w[sl]
    m = (ws > 0) & np.isfinite(xs) & np.isfinite(ys)
    return xs[m], ys[m], ws[m]


def learn_edges(x, y, w, warm, bx, by, rule):
    xe = [edges_from(rule, bx, x[:warm, j], w[:warm]) for j in range(x.shape[1])]
    ye = [edges_from(rule, by, y[:warm, t], w[:warm]) for t in range(y.shape[1])]
    return xe, ye


# 1, 2. Plug-in MI against scikit-learn; Miller-Madow against its definition.

MI_GRID = [  # (rule, bx, by, window)
    ("fixed", 4, 4, None),
    ("quantile", 16, 8, None),
    ("quantile", 16, 16, 400),
    ("fixed", 32, 8, 400),
    ("quantile", 32, 32, 2000),
]


def mm_longhand(a: np.ndarray, b: np.ndarray) -> float:
    """Miller-Madow MI by hand: ``H_MM = H + (m - 1)/(2N)`` on each entropy, plain Python."""
    n = len(a)

    def h_mm(keys) -> float:
        c = Counter(keys)
        h = -math.fsum((v / n) * math.log(v / n) for v in c.values())
        return h + (len(c) - 1) / (2 * n)

    return (
        h_mm(a.tolist()) + h_mm(b.tolist()) - h_mm(list(zip(a.tolist(), b.tolist(), strict=True)))
    )


def check_mi(rule, bx, by, window, n=4000, read_every=97, seed=3) -> dict[str, float]:
    x, y, w = stream(n, 4, 2, seed=seed)
    xe, ye = learn_edges(x, y, w, 500, bx, by, rule)
    m = BinnedMI(xe, ye, window=window)
    res = {"sklearn": 0.0, "sklearn_rel": 0.0, "mm_def": 0.0, "mm_plus_sign": 0.0, "reads": 0}
    for i in range(n):
        if i % read_every == 0 and i > 0:
            r = m.read()
            for j in range(4):
                for t in range(2):
                    xs, ys, _ = pair_sample(x, y, w, i, window, j, t)
                    if len(xs) < 2:
                        continue
                    a, b = labels(xs, xe[j]), labels(ys, ye[t])
                    ref = mutual_info_score(a, b)
                    gap = abs(r["mi"][j, t] - ref)
                    res["sklearn"] = max(res["sklearn"], gap)
                    if ref > 1e-3:
                        res["sklearn_rel"] = max(res["sklearn_rel"], gap / ref)
                    mm = mm_longhand(a, b)
                    res["mm_def"] = max(res["mm_def"], abs(r["mi_mm"][j, t] - mm))
                    # The correction added, as the request wrote it, against the definition.
                    plus = ref + (r["b_xy"][j, t] - r["b_x"][j, t] - r["b_y"][j, t] + 1) / (
                        2 * len(xs)
                    )
                    res["mm_plus_sign"] = max(res["mm_plus_sign"], abs(plus - mm))
                    res["reads"] += 1
        m.update(x[i], y[i], w[i])
    return res


# 3. The Gaussian band.


def gauss_legendre_cells(xe: np.ndarray, ye: np.ndarray, rho: float, nodes: int = 96) -> np.ndarray:
    """Cell probabilities of a standard bivariate normal on the grid ``xe`` by ``ye``:
    ``P(a, b) = int_{x in a} phi(x) [Phi((y_hi - rho x)/s) - Phi((y_lo - rho x)/s)] dx``."""
    s = math.sqrt(1.0 - rho * rho)
    gx, gw = np.polynomial.legendre.leggauss(nodes)
    xb = np.concatenate([[-12.0], xe, [12.0]])
    yb = np.concatenate([[-np.inf], ye, [np.inf]])
    p = np.zeros((len(xb) - 1, len(yb) - 1))
    for a in range(len(xb) - 1):
        lo, hi = xb[a], xb[a + 1]
        xs = 0.5 * (hi - lo) * gx + 0.5 * (hi + lo)
        ws = 0.5 * (hi - lo) * gw * np.exp(-0.5 * xs * xs) / math.sqrt(2 * math.pi)
        cdf = ndtr((yb[None, :] - rho * xs[:, None]) / s)
        p[a] = ws @ np.diff(cdf, axis=1)
    return p


def binned_mi_population(p: np.ndarray) -> float:
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / (pa * pb)[m])))


def check_gaussian(
    rho: float, b: int, n: int = 20_000, seeds: int = 40, warm: int = 2000
) -> dict[str, float]:
    """Plug-in and Miller-Madow MI of a bivariate normal on ``b x b`` quantile bins learned
    from the first ``warm`` rows; the table of each seed's ``n`` rows is read by the
    model's own read (``mi_from_tables``), which check 1 holds to scikit-learn."""
    x0, y0 = gaussian(warm, rho, seed=10_000)
    xe = edges_from("quantile", b, x0[:, 0])
    ye = edges_from("quantile", b, y0[:, 0])
    i_true = -0.5 * math.log(1.0 - rho * rho)
    i_bin = binned_mi_population(gauss_legendre_cells(xe, ye, rho))
    plug, mm = [], []
    for s in range(seeds):
        x, y = gaussian(n, rho, seed=s)
        c = np.zeros((len(xe) + 1, len(ye) + 1))
        np.add.at(c, (labels(x[:, 0], xe), labels(y[:, 0], ye)), 1.0)
        r = mi_from_tables(c)
        plug.append(float(r["mi"]))
        mm.append(float(r["mi_mm"]))
    plug_a, mm_a = np.array(plug), np.array(mm)
    sd = float(mm_a.std(ddof=1))
    return {
        "rho": rho,
        "bins": b,
        "I": i_true,
        "I_bin": i_bin,
        "loss": i_true - i_bin,
        "plugin": float(plug_a.mean()),
        "mm": float(mm_a.mean()),
        "sd": sd,
        "bias_pred": (b - 1) ** 2 / (2 * n),
        "mm_z": float((mm_a.mean() - i_bin) / (sd / math.sqrt(seeds))),
        "mm_worst": float(np.max(np.abs(mm_a - i_bin))),
    }


# 4. Decay against a plain-Python longhand.


def check_decay(
    half_life: float, n: int = 6000, seed: int = 4, bx: int = 8, by: int = 6
) -> dict[str, float]:
    x, y, w = stream(n, 3, 2, seed=seed, weights="float")
    xe, ye = learn_edges(x, y, w, 400, bx, by, "quantile")
    m = BinnedMI(xe, ye, half_life=half_life)
    lam = 0.5 ** (1.0 / half_life)
    k, t = 3, 2
    nbx, nby = m.bx, m.by
    hand = [[[[0.0] * nby for _ in range(nbx)] for _ in range(t)] for _ in range(k)]
    worst_tab = worst_mi = worst_ent = 0.0
    folds = 0
    for i in range(n):
        if i > 0:
            for pj in hand:
                for pt in pj:
                    for row in pt:
                        for c in range(nby):
                            row[c] *= lam
        wi = float(w[i])
        if wi > 0:
            for j in range(k):
                if not math.isfinite(x[i, j]):
                    continue
                a = int(sum(1 for e in xe[j] if e <= x[i, j]))
                for tt in range(t):
                    if math.isfinite(y[i, tt]):
                        b = int(sum(1 for e in ye[tt] if e <= y[i, tt]))
                        hand[j][tt][a][b] += wi
        before = m.scale
        m.update(x[i], y[i], w[i])
        folds += m.scale > before
        if i % 250 == 249 or i == n - 1:
            tab = m.tables()
            ref = np.array(hand)
            worst_tab = max(worst_tab, float(np.max(np.abs(tab - ref)) / np.max(ref)))
            r = m.read()
            for j in range(k):
                for tt in range(t):
                    c = hand[j][tt]  # plain-Python MI from the longhand table
                    tot = math.fsum(v for row in c for v in row)
                    ra = [math.fsum(row) for row in c]
                    cb = [math.fsum(c[a][b] for a in range(nbx)) for b in range(nby)]
                    mi = math.fsum(
                        (c[a][b] / tot) * math.log(c[a][b] * tot / (ra[a] * cb[b]))
                        for a in range(nbx)
                        for b in range(nby)
                        if c[a][b] > 0
                    )
                    worst_mi = max(worst_mi, abs(r["mi"][j, tt] - mi))
                    # scipy's entropy on the weighted margins and joint (the scan's decayed oracle)
                    cj = np.array(c)
                    ent = (
                        stats.entropy(cj.sum(1))
                        + stats.entropy(cj.sum(0))
                        - stats.entropy(cj.ravel())
                    )
                    worst_ent = max(worst_ent, abs(r["mi"][j, tt] - ent))
    return {
        "half_life": half_life,
        "rows": n,
        "folds": folds,
        "table_rel": worst_tab,
        "mi_longhand": worst_mi,
        "mi_scipy_entropy": worst_ent,
    }


# 5. The rings: exactly the rows they should hold.


def check_rings(window: int = 60, n: int = 2000, seed: int = 5) -> dict[str, float]:
    x, y, w = stream(n, 3, 2, seed=seed, first_zero=True)
    w[300 : 300 + 2 * window] = 0.0  # a run of zero weights longer than the window
    x[900 : 900 + window // 2] = np.nan  # a gap of half a window
    rw = RankWindow(3, 2, window, rank_every=1)
    xe, ye = learn_edges(x, y, np.where(w > 0, 1.0, 0.0), 200, 6, 5, "quantile")
    mi = BinnedMI(xe, ye, window=window)
    bad_rank = bad_pairs = bad_mi = 0
    empty_checked = False
    for i in range(n):
        for j in range(3):
            for t in range(2):
                xs, ys, _ = pair_sample(x, y, w, i, window, j, t)
                gx, gy = rw.sample(j, t)
                bad_rank += not (np.array_equal(gx, xs) and np.array_equal(gy, ys))
                bad_pairs += rw.n[j, t] != len(xs) * (len(xs) - 1) // 2
        tab = mi.tables()
        ref = np.zeros_like(tab)
        sl = window_rows(i, window)
        for r in range(sl.start, sl.stop):
            if w[r] > 0:
                for j in range(3):
                    for t in range(2):
                        if np.isfinite(x[r, j]) and np.isfinite(y[r, t]):
                            ref[j, t, labels(x[r, j], xe[j]), labels(y[r, t], ye[t])] += w[r]
        bad_mi += not np.array_equal(tab, ref)
        if i == 300 + 2 * window:  # the window has held only zero-weight rows
            empty_checked = bool(np.all(tab == 0.0) and np.all(np.isnan(mi.read()["mi"])))
            empty_checked &= bool(np.all(rw.n == 0) and np.all(np.isnan(rw.kendall())))
        rw.learn(x[i], y[i], float(w[i]))
        mi.update(x[i], y[i], w[i])
    return {
        "rows": n,
        "rank_mismatch": bad_rank,
        "kendall_pairs_mismatch": bad_pairs,
        "mi_table_mismatch": bad_mi,
        "empty_window_ok": int(empty_checked),
        "first_row_zero": int(w[0] == 0),
    }


def check_window_float(window: int = 500, n: int = 100_000, seed: int = 6) -> dict[str, float]:
    """Fractional weights: the subtraction leaves residue; how much, and does it move MI."""
    x, y, w = stream(n, 1, 1, seed=seed, weights="float", ties=False)
    xe, ye = learn_edges(x, y, w, 500, 16, 16, "quantile")
    m = BinnedMI(xe, ye, window=window)
    for i in range(n):
        m.update(x[i], y[i], w[i])
    xs, ys, ws = pair_sample(x, y, w, n, window, 0, 0)
    ref = np.zeros((m.bx, m.by))
    np.add.at(ref, (labels(xs, xe[0]), labels(ys, ye[0])), ws)
    tab = m.tables()[0, 0]
    rr = mi_from_tables(ref)
    r = m.read()
    return {
        "cell_residue": float(np.max(np.abs(tab - ref))),
        "false_nonzero": int(np.sum((tab != 0) & (ref == 0))),
        "mi_gap": float(abs(r["mi"][0, 0] - rr["mi"])),
        "mi_entropy_gap": float(
            abs(
                r["mi"][0, 0]
                - (
                    stats.entropy(ref.sum(1))
                    + stats.entropy(ref.sum(0))
                    - stats.entropy(ref.ravel())
                )
            )
        ),
    }


# 6. Ranks against scipy.


def xi_longhand(xs: np.ndarray, ys: np.ndarray) -> float:
    """Chatterjee's eq. 1.2 in plain Python, ties in x broken by arrival (a stable sort)."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])  # Python's sort is stable
    yo = [ys[i] for i in order]
    r = [sum(1 for v in yo if v <= yi) for yi in yo]
    l_ = [sum(1 for v in yo if v >= yi) for yi in yo]
    num = sum(abs(r[i + 1] - r[i]) for i in range(n - 1))
    den = 2 * sum(li * (n - li) for li in l_)
    return 1.0 - n * num / den


def check_ranks(
    window: int, rank_every: int, n: int = 1500, seed: int = 7, chunks: int = 13
) -> dict[str, float]:
    x, y, w = stream(n, 4, 2, seed=seed)
    out = run_ranks(RankWindow(4, 2, window, rank_every), x, y, w)
    same = run_ranks(RankWindow(4, 2, window, rank_every), x, y, w, chunks=chunks)
    chunk_ok = all(np.array_equal(out[key], same[key], equal_nan=True) for key in out)
    res = {
        "spearman": 0.0,
        "kendall": 0.0,
        "xi_scipy": 0.0,
        "xi_longhand": 0.0,
        "xi_cont": 0.0,
        "xi_scipy_tied_x": 0.0,
        "held_ok": 1,
        "chunks_ok": int(chunk_ok),
        "reads": 0,
    }
    last_read = 0
    rw = RankWindow(4, 2, window, rank_every)
    for i in range(n):
        read = i % rank_every == 0
        if read:
            last_read = i
        for j in range(4):
            for t in range(2):
                xs, ys, _ = pair_sample(x, y, w, i, window, j, t)
                if len(xs) >= 3:
                    kt = stats.kendalltau(xs, ys).statistic
                    if np.isfinite(kt):
                        res["kendall"] = max(res["kendall"], abs(out["kendall"][i, j, t] - kt))
                if not read:
                    res["held_ok"] &= int(
                        np.array_equal(
                            out["spearman"][i, j, t],
                            out["spearman"][last_read, j, t],
                            equal_nan=True,
                        )
                    )
                    continue
                if len(xs) < 3:
                    continue
                res["reads"] += 1
                sp = stats.spearmanr(xs, ys).statistic
                if np.isfinite(sp):
                    res["spearman"] = max(res["spearman"], abs(out["spearman"][i, j, t] - sp))
                xi = out["xi"][i, j, t]
                if len(np.unique(xs)) == len(xs):
                    res["xi_scipy"] = max(
                        res["xi_scipy"], abs(xi - stats.chatterjeexi(xs, ys).statistic)
                    )
                else:
                    res["xi_scipy_tied_x"] = max(
                        res["xi_scipy_tied_x"], abs(xi - stats.chatterjeexi(xs, ys).statistic)
                    )
                if i % (7 * rank_every) == 0:
                    res["xi_longhand"] = max(
                        res["xi_longhand"], abs(xi - xi_longhand(xs.tolist(), ys.tolist()))
                    )
        if read and i > 0:
            # y_continuous: the model's other form, on every pair at the read rows
            for j in range(4):
                for t in range(2):
                    xs, ys = rw.sample(j, t)
                    if len(xs) >= 3 and len(np.unique(xs)) == len(xs):
                        a = chatterjee_xi(xs, ys, y_continuous=True)
                        b = stats.chatterjeexi(xs, ys, y_continuous=True).statistic
                        res["xi_cont"] = max(res["xi_cont"], abs(a - b))
        rw.learn(x[i], y[i], float(w[i]))
    return res


# 7. Cost.


def time_ranks(
    window: int, k: int = 10, t: int = 1, rows: int = 200, reads: int = 5, seed: int = 8
) -> dict[str, float]:
    x, y, w = stream(window + rows, k, t, seed=seed, gaps=False, ties=False, weights="ones")
    rw = RankWindow(k, t, window, rank_every=10**9)
    t0 = time.perf_counter()
    for i in range(window):
        rw.learn(x[i], y[i], 1.0)
    fill = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(window, window + rows):
        rw.learn(x[i], y[i], 1.0)
    per_row = (time.perf_counter() - t0) / rows
    t0 = time.perf_counter()
    for _ in range(reads):
        rw.read_ranks()
    per_read = (time.perf_counter() - t0) / reads
    t0 = time.perf_counter()
    for _ in range(1000):
        rw.kendall()
    kendall_read = (time.perf_counter() - t0) / 1000
    return {
        "W": window,
        "pairs": k * t,
        "fill_s": fill,
        "row_us": per_row * 1e6,
        "read_ms": per_read * 1e3,
        "kendall_read_us": kendall_read * 1e6,
    }


def time_mi(
    window: int | None, k: int = 10, t: int = 1, bins: int = 16, rows: int = 3000, seed: int = 9
) -> dict[str, float]:
    fill = window or 0
    x, y, w = stream(fill + rows, k, t, seed=seed, gaps=False, ties=False, weights="ones")
    xe, ye = learn_edges(x, y, w, 500, bins, bins, "quantile")
    m = BinnedMI(xe, ye, window=window)
    for i in range(fill):  # a full ring, so every timed row also removes one
        m.update(x[i], y[i], 1.0)
    t0 = time.perf_counter()
    for i in range(fill, fill + rows):
        m.update(x[i], y[i], 1.0)
    per_row = (time.perf_counter() - t0) / rows
    t0 = time.perf_counter()
    for _ in range(200):
        m.read()
    per_read = (time.perf_counter() - t0) / 200
    return {
        "W": window or 0,
        "pairs": k * t,
        "bins": bins,
        "row_us": per_row * 1e6,
        "read_us": per_read * 1e6,
    }


# 8. State size.


def mi_state_bytes(k: int, t: int, bx: int, by: int, window: int | None) -> int:
    """``8 k T bx by`` (table) + ``8 (k (bx-1) + T (by-1))`` (edges) + 24 (lam, scale, rows);
    a window adds ``2 W (k + T)`` (uint16 bins) + ``8 W`` (weights) + 8 (head)."""
    b = 8 * k * t * bx * by + 8 * (k * (bx - 1) + t * (by - 1)) + 24
    if window:
        b += 2 * window * (k + t) + 8 * window + 8
    return b


def rank_state_bytes(k: int, t: int, window: int) -> int:
    """``8 W (k + T)`` (values) + ``8 W`` (weights) + ``8 * 4 k T`` (S, n_0, n_1, n_2)
    + ``8 * 2 k T`` (held Spearman, xi) + 16 (head, rows)."""
    return 8 * window * (k + t) + 8 * window + 32 * k * t + 16 * k * t + 16


def check_state(k: int = 10, t: int = 1, bins: int = 16, window: int = 1000) -> dict[str, int]:
    x, y, w = stream(3000, k, t, seed=11, gaps=False, ties=False, weights="ones")
    xe, ye = learn_edges(x, y, w, 2000, bins, bins, "quantile")
    out = {}
    for win in (None, window):
        m = BinnedMI(xe, ye, window=win)
        for i in range(1200):
            m.update(x[i], y[i], 1.0)
        counted = sum(a.nbytes for a in m.state_arrays().values())
        out["mi_" + ("window" if win else "decay")] = counted
        out["mi_" + ("window" if win else "decay") + "_formula"] = mi_state_bytes(
            k, t, bins, bins, win
        )
    rw = RankWindow(k, t, window)
    for i in range(1200):
        rw.learn(x[i], y[i], 1.0)
    out["ranks"] = sum(a.nbytes for a in rw.state_arrays().values())
    out["ranks_formula"] = rank_state_bytes(k, t, window)
    return out


# Numerical risk: MI near independence against an extended-precision read.


def check_mi_precision(n: int = 200_000, bins: int = 32, seed: int = 12) -> dict[str, float]:
    """MI of nearly independent columns (true MI 0, plug-in about (b-1)^2/(2N)):
    the double read against the same sum in ``np.longdouble``."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, bins, n)
    b = (a * (rng.random(n) < 0.002) + rng.integers(0, bins, n) * 1) % bins
    c = np.zeros((bins, bins))
    np.add.at(c, (a, b), 1.0)
    r = float(mi_from_tables(c)["mi"])
    cl = c.astype(np.longdouble)
    nl = cl.sum()
    pa, pb = cl.sum(1, keepdims=True), cl.sum(0, keepdims=True)
    m = cl > 0
    ref = (cl[m] / nl * np.log(cl[m] * nl / (pa * pb)[m])).sum()
    ent = stats.entropy(c.sum(1)) + stats.entropy(c.sum(0)) - stats.entropy(c.ravel())
    return {
        "mi": r,
        "abs_gap": float(abs(r - ref)),
        "rel_gap": float(abs(r - ref) / ref),
        "entropy_form_gap": float(abs(ent - ref)),
        "sklearn_gap": float(abs(mutual_info_score(a, b) - ref)),
        "longdouble_eps": float(np.finfo(np.longdouble).eps),
    }
