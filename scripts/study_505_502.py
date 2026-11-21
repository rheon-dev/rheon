"""Study 505 / 502: block incremental PCA (part 1) and kernel RLS (part 2).

Python and numpy only; nothing here is part of the package. Every input is drawn
from seeded generators; no data file is read or written. The notes are
docs/records/STUDY-505-502-PCA-KRLS.md.

Part 1 (task 505): a block incremental PCA on an exponentially weighted
covariance, in two forms, both in the mean form `ew_cov` keeps:

* exact:     (W, m, C), C the k x k weighted covariance, folded block by block
             (Chan's pairwise combination with a decay), read by `eigh`;
* truncated: (W, m, V, s, tr), V the top-r basis (r x k), s its singular values
             in the mean form (C ~= V' diag(s^2) V) and tr = trace(C), updated
             by the SVD of a stacked (r + b + 1) x k matrix (Ross et al. 2008,
             the form scikit-learn's IncrementalPCA runs, with a decay added).

Oracles: sklearn IncrementalPCA.partial_fit on the same batches (lambda = 1,
unit weights, the only case it covers), and sklearn PCA(svd_solver='full') on
the decay-weighted rows. PCA takes no sample weights and re-centres by the
unweighted mean, so it is given the rows Y = sqrt(u_i / U) (x_i - m_u) stacked
with -Y: the stack's unweighted mean is exactly 0, so PCA's centring is a
no-op, and its scatter is 2 Y'Y = 2 C, so the components and the ratios are
those of the weighted covariance C. scipy.linalg.eigh on np.cov(aweights=u) is
the second oracle. No randomized solver is used anywhere.

Part 2 (task 502): kernel recursive least squares with a capped dictionary
(Engel, Mannor and Meir 2004, approximate linear dependence, with a ridge r),
and the sliding-window KRLS. Oracles: sklearn KernelRidge on the dictionary
atoms and on the window rows, and sklearn Nystroem(landmarks = the dictionary)
then Ridge on every learned row (the subset-of-regressors fit KRLS runs).

Run: uv run python scripts/study_505_502.py pca    (about 20 s)
     uv run python scripts/study_505_502.py krls   (about 150 s; run with nohup)
     uv run python scripts/study_505_502.py krls exact zero cost   (any subset of:
         exact grid window zero numerics cost)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.linalg as sla
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

B = 200  # rows per block (cut on row counts, never on chunks)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def make_stream(seed: int, n: int, k: int, offset: float = 5.0) -> np.ndarray:
    """Rows with a separated, geometric spectrum (0.6^j), a mean offset, and a
    regime change half way (a new random rotation and mean), so that the decay
    has something to forget."""
    rng = np.random.default_rng(seed)
    spec = np.sqrt(0.6 ** np.arange(k)) * 3.0
    out = np.empty((n, k))
    half = n // 2
    for lo, hi in ((0, half), (half, n)):
        q, _ = np.linalg.qr(rng.standard_normal((k, k)))
        mu = offset + rng.standard_normal(k)
        z = rng.standard_normal((hi - lo, k))
        out[lo:hi] = mu + (z * spec) @ q.T
    return out


# --------------------------------------------------------------------------
# block contributions (what one block of rows adds, decayed to the block end)
# --------------------------------------------------------------------------


def block_part(x: np.ndarray, w: np.ndarray, lam: float):
    """Each row's weight at the end of its block u_i = w_i lam^(b-1-i), the
    block's weight U, weighted mean m_B and centred rows a_i = x_i - m_B.
    U == 0 (an all-zero-weight block) returns None for the mean."""
    b = len(w)
    u = w * lam ** np.arange(b - 1, -1, -1)
    big_u = u.sum()
    if big_u == 0.0:
        return u, 0.0, None, None
    m_b = (u @ x) / big_u
    return u, big_u, m_b, x - m_b


class ExactEw:
    """Mean form, as `ew_cov`: W, m, C (k x k).  Per block:
    W' = lam^b W + U,  a = lam^b W / W',  beta = U / W'
    m' = m + beta (m_B - m)
    C' = a C + beta C_B + a beta (m - m_B)(m - m_B)'.
    An all-zero block (U = 0) leaves m and C untouched and only decays W."""

    def __init__(self, k: int):
        self.w = 0.0
        self.m = np.zeros(k)
        self.c = np.zeros((k, k))

    def update(self, x, wt, lam):
        b = len(wt)
        u, big_u, m_b, a_rows = block_part(x, wt, lam)
        w_old = lam**b * self.w
        if big_u == 0.0:
            self.w = w_old
            return
        w_new = w_old + big_u
        a = w_old / w_new
        beta = big_u / w_new
        c_b = (a_rows * (u / big_u)[:, None]).T @ a_rows
        d = self.m - m_b
        self.c = a * self.c + beta * c_b + (a * beta) * np.outer(d, d)
        self.m = self.m + beta * (m_b - self.m)
        self.w = w_new

    def remove(self, x, wt_now):
        """Take out rows whose weight *now* is wt_now (the window's leaving
        rows): the inverse of the combination above."""
        big_u = wt_now.sum()
        if big_u == 0.0:
            return
        m_b = (wt_now @ x) / big_u
        a_rows = x - m_b
        c_b = (a_rows * (wt_now / big_u)[:, None]).T @ a_rows
        w_r = self.w - big_u
        m_r = (self.w * self.m - big_u * m_b) / w_r
        d = m_r - m_b
        # W C = W_r C_r + U C_B + (W_r U / W) d d'
        self.c = (self.w * self.c - big_u * c_b - (w_r * big_u / self.w) * np.outer(d, d)) / w_r
        self.m = m_r
        self.w = w_r

    def decay_only(self, lam, b):
        self.w *= lam**b

    def top(self, n):
        vals, vecs = sla.eigh(self.c)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        return vecs[:, :n].T, vals[:n], vals[:n] / np.trace(self.c)


class TruncatedEw:
    """Mean form of the block incremental SVD: W, m, V (r x k), s (r), tr.
    Per block, the SVD of the stack
        [ sqrt(a) diag(s) V ;  sqrt(beta u_i / U) a_i (b rows) ;
          sqrt(a beta) (m - m_B) ]
    keeps its top r; tr' = a tr + beta tr(C_B) + a beta |m - m_B|^2 exactly.
    At a = 0 (first block) the stack is the block alone; at lam = 1 and unit
    weights this is scikit-learn's IncrementalPCA.partial_fit scaled by 1/W."""

    def __init__(self, k: int, r: int):
        self.r = r
        self.w = 0.0
        self.m = np.zeros(k)
        self.v = np.zeros((0, k))
        self.s = np.zeros(0)
        self.tr = 0.0

    def update(self, x, wt, lam):
        b = len(wt)
        u, big_u, m_b, a_rows = block_part(x, wt, lam)
        w_old = lam**b * self.w
        if big_u == 0.0:
            self.w = w_old
            return
        w_new = w_old + big_u
        a = w_old / w_new
        beta = big_u / w_new
        d = self.m - m_b
        rows = [np.sqrt(beta * u / big_u)[:, None] * a_rows]
        if a > 0.0:
            rows = [np.sqrt(a) * self.s[:, None] * self.v, *rows, np.sqrt(a * beta) * d[None, :]]
        _, s, vt = sla.svd(np.vstack(rows), full_matrices=False, check_finite=False)
        self.v = vt[: self.r]
        self.s = s[: self.r]
        tr_b = float(((a_rows**2).sum(axis=1) * (u / big_u)).sum())
        self.tr = a * self.tr + beta * tr_b + a * beta * float(d @ d)
        self.m = self.m + beta * (m_b - self.m)
        self.w = w_new

    def top(self, n):
        return self.v[:n], self.s[:n] ** 2, self.s[:n] ** 2 / self.tr


# --------------------------------------------------------------------------
# oracles and comparisons
# --------------------------------------------------------------------------


def final_weights(wt: np.ndarray, lam: float) -> np.ndarray:
    """Each row's weight at the end of the stream, w_i lam^(N-1-i)."""
    n = len(wt)
    return wt * lam ** np.arange(n - 1, -1, -1)


def oracle_pca(x: np.ndarray, u: np.ndarray, n: int):
    """sklearn PCA(svd_solver='full') on [Y; -Y], Y = sqrt(u/U)(x - m_u)."""
    keep = u > 0
    x, u = x[keep], u[keep]
    m_u = (u @ x) / u.sum()
    y = np.sqrt(u / u.sum())[:, None] * (x - m_u)
    p = PCA(n_components=n, svd_solver="full").fit(np.vstack([y, -y]))
    return p.components_, p.explained_variance_ratio_


def oracle_eigh(x: np.ndarray, u: np.ndarray, n: int):
    """scipy eigh of numpy's weighted covariance (aweights, biased)."""
    keep = u > 0
    c = np.cov(x[keep], rowvar=False, aweights=u[keep], bias=True)
    c = np.atleast_2d(c)
    vals, vecs = sla.eigh(c)
    order = np.argsort(vals)[::-1]
    return vecs[:, order[:n]].T, vals[order[:n]] / np.trace(c)


def compare(v1, r1, v2, r2):
    """(largest principal angle, worst 1 - |<v1_i, v2_j>| after matching by the
    absolute inner product, largest explained-variance-ratio gap)."""
    ang = float(np.max(sla.subspace_angles(v1.T, v2.T)))
    g = np.abs(v1 @ v2.T)
    match = g.argmax(axis=1)
    cos_gap = float(np.max(1.0 - g[np.arange(len(match)), match]))
    ev = float(np.max(np.abs(np.asarray(r1) - np.asarray(r2)[match])))
    return ang, cos_gap, ev


def n_compare(k: int) -> int:
    return {2: 1, 5: 2, 20: 5}.get(k, max(1, k // 4))


def run_stream(model, x, wt, lam):
    for lo in range(0, len(x), B):
        model.update(x[lo : lo + B], wt[lo : lo + B], lam)
    return model


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

KS = (2, 5, 20)
LAMS = (1.0, 0.999, 0.99)
NS = (2000, 20000)
SEEDS = range(5)


def fmt(v: float) -> str:
    return f"{v:.1e}"


def exp_ipca():
    """lambda = 1, unit weights, against IncrementalPCA on the same batches."""
    print("\n### lambda = 1, unit weights: against IncrementalPCA, same 200-row batches\n")
    print(
        "| k | n compared | r | N | exact vs IPCA(all): angle, 1-cos, ratio | "
        "truncated(r) vs IPCA(r): angle, 1-cos, ratio | IPCA(r) vs PCA: angle |"
    )
    print("|---|---|---|---|---|---|---|")
    worst = np.zeros(3)
    worst_t = np.zeros(3)
    for k in KS:
        nc = n_compare(k)
        r = min(k, nc + 2)
        for n in NS:
            acc_e = np.zeros(3)
            acc_t = np.zeros(3)
            acc_l = 0.0
            for seed in SEEDS:
                x = make_stream(seed, n, k)
                wt = np.ones(n)
                ex = run_stream(ExactEw(k), x, wt, 1.0)
                tr = run_stream(TruncatedEw(k, r), x, wt, 1.0)
                full = IncrementalPCA(n_components=None)
                part = IncrementalPCA(n_components=r)
                for lo in range(0, n, B):
                    full.partial_fit(x[lo : lo + B])
                    part.partial_fit(x[lo : lo + B])
                ve, _, re_ = ex.top(nc)
                acc_e = np.maximum(
                    acc_e,
                    compare(ve, re_, full.components_[:nc], full.explained_variance_ratio_[:nc]),
                )
                vt, _, rt = tr.top(nc)
                acc_t = np.maximum(
                    acc_t,
                    compare(vt, rt, part.components_[:nc], part.explained_variance_ratio_[:nc]),
                )
                pc = PCA(n_components=nc, svd_solver="full").fit(x)
                acc_l = max(
                    acc_l,
                    compare(
                        part.components_[:nc],
                        part.explained_variance_ratio_[:nc],
                        pc.components_,
                        pc.explained_variance_ratio_,
                    )[0],
                )
            worst = np.maximum(worst, acc_e)
            worst_t = np.maximum(worst_t, acc_t)
            print(
                f"| {k} | {nc} | {r} | {n} | {', '.join(map(fmt, acc_e))} | "
                f"{', '.join(map(fmt, acc_t))} | {fmt(acc_l)} |"
            )
    print(
        f"\nworst exact vs IPCA(all): angle {fmt(worst[0])}, 1-cos {fmt(worst[1])}, "
        f"ratio gap {fmt(worst[2])}"
    )
    print(
        f"worst truncated vs IPCA(r): angle {fmt(worst_t[0])}, 1-cos {fmt(worst_t[1])}, "
        f"ratio gap {fmt(worst_t[2])}"
    )


def exp_decay():
    """Decay and random weights (one row in 10 at zero weight), against PCA on
    the decay-weighted rows."""
    print("\n### Decay, weights U(0.5, 1.5) with 10% zeros: against PCA on the weighted rows\n")
    print(
        "| k | n | lambda | N | exact vs PCA: angle, 1-cos, ratio | exact vs eigh: angle | "
        "truncated r=n+2 vs PCA: angle, ratio | truncated r=k vs PCA: angle, ratio |"
    )
    print("|---|---|---|---|---|---|---|---|")
    worst = np.zeros(3)
    worst_rk = np.zeros(2)
    for k in KS:
        nc = n_compare(k)
        r = min(k, nc + 2)
        for lam in LAMS:
            for n in NS:
                acc = np.zeros(3)
                acc_h = 0.0
                acc_t = np.zeros(2)
                acc_k = np.zeros(2)
                for seed in SEEDS:
                    x = make_stream(seed, n, k)
                    rng = np.random.default_rng(1000 + seed)
                    wt = rng.uniform(0.5, 1.5, n) * (rng.random(n) >= 0.1)
                    u = final_weights(wt, lam)
                    vo, ro = oracle_pca(x, u, nc)
                    vh, rh = oracle_eigh(x, u, nc)
                    ex = run_stream(ExactEw(k), x, wt, lam)
                    ve, _, re_ = ex.top(nc)
                    acc = np.maximum(acc, compare(ve, re_, vo, ro))
                    acc_h = max(acc_h, compare(ve, re_, vh, rh)[0])
                    tr = run_stream(TruncatedEw(k, r), x, wt, lam)
                    vt, _, rt = tr.top(nc)
                    c = compare(vt, rt, vo, ro)
                    acc_t = np.maximum(acc_t, (c[0], c[2]))
                    tk = run_stream(TruncatedEw(k, k), x, wt, lam)
                    vk, _, rk = tk.top(nc)
                    c = compare(vk, rk, vo, ro)
                    acc_k = np.maximum(acc_k, (c[0], c[2]))
                if lam < 1.0:
                    worst = np.maximum(worst, acc)
                    worst_rk = np.maximum(worst_rk, acc_k)
                print(
                    f"| {k} | {nc} | {lam} | {n} | {', '.join(map(fmt, acc))} | {fmt(acc_h)} | "
                    f"{', '.join(map(fmt, acc_t))} | {', '.join(map(fmt, acc_k))} |"
                )
    print(
        f"\nworst (lambda < 1) exact vs weighted PCA: angle {fmt(worst[0])}, "
        f"1-cos {fmt(worst[1])}, ratio gap {fmt(worst[2])}"
    )
    print(
        f"worst (lambda < 1) truncated r = k vs weighted PCA: angle {fmt(worst_rk[0])}, "
        f"ratio gap {fmt(worst_rk[1])}"
    )


def exp_window(width: int = 2000):
    """Exact form with a window of the last `width` rows: each block added, the
    block that left removed at its current decayed weight. The truncated form
    has no window (an SVD cannot give back a row it has folded in)."""
    print(f"\n### Window of the last {width} rows (N = 20000): against PCA on the window rows\n")
    print("| k | n | lambda | angle | 1-cos | ratio gap |")
    print("|---|---|---|---|---|---|")
    n = 20000
    worst = np.zeros(3)
    for k in KS:
        nc = n_compare(k)
        for lam in (1.0, 0.999):
            acc = np.zeros(3)
            for seed in SEEDS:
                x = make_stream(seed, n, k)
                rng = np.random.default_rng(2000 + seed)
                wt = rng.uniform(0.5, 1.5, n) * (rng.random(n) >= 0.1) if lam < 1 else np.ones(n)
                ex = ExactEw(k)
                for lo in range(0, n, B):
                    hi = lo + B
                    ex.update(x[lo:hi], wt[lo:hi], lam)
                    out_lo, out_hi = lo - width, hi - width
                    if out_hi > 0:
                        idx = np.arange(max(out_lo, 0), out_hi)
                        ex.remove(x[idx], wt[idx] * lam ** (hi - 1 - idx))
                    # read at the end of every 10th block, against the oracle
                    if (hi // B) % 10 == 0 and hi >= width:
                        xs, ws = x[hi - width : hi], wt[hi - width : hi]
                        u = ws * lam ** np.arange(width - 1, -1, -1)
                        if lam == 1.0:
                            p = PCA(n_components=nc, svd_solver="full").fit(xs)
                            vo, ro = p.components_, p.explained_variance_ratio_
                        else:
                            vo, ro = oracle_pca(xs, u, nc)
                        ve, _, re_ = ex.top(nc)
                        acc = np.maximum(acc, compare(ve, re_, vo, ro))
            worst = np.maximum(worst, acc)
            print(f"| {k} | {nc} | {lam} | {', '.join(map(fmt, acc)).replace(', ', ' | ')} |")
    print(
        f"\nworst window: angle {fmt(worst[0])}, 1-cos {fmt(worst[1])}, ratio gap {fmt(worst[2])}"
    )


def exp_zero_weight():
    """A stream whose first 300 rows (the first block and half the second) and
    two later whole blocks are at zero weight."""
    print("\n### Zero-weight rows\n")
    k, n, lam = 5, 4000, 0.999
    nc = 2
    x = make_stream(7, n, k)
    wt = np.ones(n)
    wt[:300] = 0.0
    wt[1000:1400] = 0.0
    ex, tr = ExactEw(k), TruncatedEw(k, 4)
    nan_seen = False
    unchanged = True
    w_ok = True
    for lo in range(0, n, B):
        before = (ex.w, ex.m.copy(), ex.c.copy(), tr.m.copy(), tr.v.copy(), tr.s.copy(), tr.tr)
        ex.update(x[lo : lo + B], wt[lo : lo + B], lam)
        tr.update(x[lo : lo + B], wt[lo : lo + B], lam)
        nan_seen |= not (
            np.isfinite(ex.c).all()
            and np.isfinite(ex.m).all()
            and np.isfinite(tr.v).all()
            and np.isfinite(tr.s).all()
            and np.isfinite(tr.tr)
            and np.isfinite(ex.w)
        )
        if not wt[lo : lo + B].any():
            unchanged &= (
                np.array_equal(before[1], ex.m)
                and np.array_equal(before[2], ex.c)
                and np.array_equal(before[3], tr.m)
                and np.array_equal(before[4], tr.v)
                and np.array_equal(before[5], tr.s)
                and before[6] == tr.tr
            )
            w_ok &= ex.w == before[0] * lam**B and tr.w == ex.w
    u = final_weights(wt, lam)
    vo, ro = oracle_pca(x, u, nc)
    ae = compare(*ex.top(nc)[::2], vo, ro)
    at = compare(*tr.top(nc)[::2], vo, ro)
    print(f"- NaN anywhere in either state: {nan_seen}")
    print(f"- all-zero blocks leave m, C and m, V, s, tr bit-identical: {unchanged}")
    print(f"- all-zero blocks only decay W (W' == lam^b W exactly, also from 0): {w_ok}")
    print(
        f"- end of stream vs PCA on the weighted rows: exact angle {fmt(ae[0])}, ratio gap "
        f"{fmt(ae[2])}; truncated (r = 4 of k = 5) angle {fmt(at[0])}, ratio gap {fmt(at[2])}"
    )


def exp_block_size():
    """The exact form's result does not depend on the block size beyond
    rounding: b = 200 against b = 1 (the per-row `ew_cov` recursion)."""
    global B
    print("\n### Block size: b = 200 against b = 1 (exact form, k = 20, N = 2000)\n")
    worst = 0.0
    for seed in SEEDS:
        for lam in LAMS:
            x = make_stream(seed, 2000, 20)
            wt = np.random.default_rng(seed).uniform(0.5, 1.5, 2000)
            saved = B
            B = 200
            a = run_stream(ExactEw(20), x, wt, lam)
            B = 1
            b1 = run_stream(ExactEw(20), x, wt, lam)
            B = saved
            worst = max(worst, float(np.max(np.abs(a.c - b1.c)) / np.max(np.abs(b1.c))))
    print(f"- largest |C(b=200) - C(b=1)| / max|C|: {fmt(worst)} (rounding, not bit-equal)")


def exp_cost():
    """Measured time per block for both forms (b = 200), and the state bytes."""
    print("\n### Cost per 200-row block (this VM, numpy's default BLAS threads, median of 5)\n")
    print(
        "| k | r | exact fold | exact eigh read | truncated fold (SVD) | "
        "exact state bytes | truncated state bytes |"
    )
    print("|---|---|---|---|---|---|---|")
    for k, r in ((20, 7), (200, 12), (2000, 12)):
        x = make_stream(0, 6 * B, k)
        wt = np.ones(len(x))
        ex, tr = ExactEw(k), TruncatedEw(k, r)
        ex.update(x[:B], wt[:B], 0.999)
        tr.update(x[:B], wt[:B], 0.999)
        te, tt, th = [], [], []
        for i in range(1, 6):
            blk = slice(i * B, (i + 1) * B)
            t0 = time.perf_counter()
            ex.update(x[blk], wt[blk], 0.999)
            t1 = time.perf_counter()
            tr.update(x[blk], wt[blk], 0.999)
            t2 = time.perf_counter()
            ex.top(r)
            t3 = time.perf_counter()
            te.append(t1 - t0)
            tt.append(t2 - t1)
            th.append(t3 - t2)
        sb_e = 8 * (1 + k + k * (k + 1) // 2)
        sb_t = 8 * (1 + k + r * k + r + 1)
        print(
            f"| {k} | {r} | {np.median(te) * 1e3:.2f} ms | {np.median(th) * 1e3:.2f} ms | "
            f"{np.median(tt) * 1e3:.2f} ms | {sb_e:,} | {sb_t:,} |"
        )


def main_pca():
    t0 = time.time()
    exp_ipca()
    exp_decay()
    exp_window()
    exp_zero_weight()
    exp_block_size()
    exp_cost()
    print(f"\n(total {time.time() - t0:.1f} s)")


# ==========================================================================
# Part 2 (task 502): kernel recursive least squares
# ==========================================================================

R_RIDGE = 0.1  # the ridge r: KernelRidge(alpha=r), Ridge(alpha=r)
NU = 1e-3  # the ALD threshold of the grid
N_ROWS = 2000
N_TEST = 500
CAPS = (20, 50, 200)
KDIMS = (2, 5)
KERNELS = ("rbf", "poly")


def kern(name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """k(a_i, b_j), shape (len(a), len(b)); gamma = 1/k for both kernels.
    rbf: exp(-gamma |x - y|^2).  poly: (gamma <x, y> + 1)^3 (scikit-learn's
    'polynomial' with degree=3, coef0=1)."""
    g = 1.0 / a.shape[1]
    if name == "rbf":
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-g * d)
    return (g * (a @ b.T) + 1.0) ** 3


def sk_kernel(name: str, k: int) -> dict:
    if name == "rbf":
        return {"kernel": "rbf", "gamma": 1.0 / k}
    return {"kernel": "polynomial", "gamma": 1.0 / k, "degree": 3, "coef0": 1.0}


def make_regression(seed: int, n: int, k: int):
    """x ~ N(0, I_k); y = sin(1.5 x.u) + 0.3 (x.v)^2 + 0.1 eps, u and v seeded
    unit vectors. The first n rows are the stream, the next N_TEST held out."""
    rng = np.random.default_rng(500 + seed)
    u = rng.standard_normal(k)
    v = rng.standard_normal(k)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    x = rng.standard_normal((n + N_TEST, k))
    y = np.sin(1.5 * x @ u) + 0.3 * (x @ v) ** 2 + 0.1 * rng.standard_normal(n + N_TEST)
    return x[:n], y[:n], x[n:], y[n:]


class AldKrls:
    """KRLS with an approximate-linear-dependence dictionary, a cap and a ridge.

    State: D (m x k, the atoms), Kinv = K_D^-1 (m x m, for the ALD test),
    P = (sum_s w_s kh_s kh_s' + r K_D)^-1 (m x m), alpha (m).  Per row x, y, w:
        kv = k(D, x),  a = Kinv kv,  delta = k(x, x) - kv' a
    admit when delta > nu and m < cap (the cap's rule: stop admitting, keep
    learning every row on the frozen dictionary); admission extends
        Kinv <- [[Kinv + a a'/delta, -a/delta], [-a'/delta, 1/delta]]
        P    <- [[P + a a'/(r delta), -a/(r delta)], [-a'/(r delta), 1/(r delta)]]
        alpha <- [alpha; 0],  kv <- [kv; k(x, x)]
    (the past rows enter the new column by their projection on the old span);
    then the weighted RLS step on the feature kv:
        g = P kv / (1/w + kv' P kv),  alpha += g (y - kv' alpha),  P -= g (P kv)'.
    w == 0 returns at once: no admission, no change.
    atoms_only=True learns only from rows it admits (the KernelRidge-on-atoms
    definition)."""

    def __init__(self, kname: str, k: int, cap: int, nu: float, r: float, atoms_only=False):
        self.kname, self.cap, self.nu, self.r = kname, cap, nu, r
        self.atoms_only = atoms_only
        self.d = np.zeros((0, k))
        self.kinv = np.zeros((0, 0))
        self.p = np.zeros((0, 0))
        self.alpha = np.zeros(0)
        self.admitted: list[int] = []
        self.neg_delta = 0
        self.min_delta_adm = np.inf

    @staticmethod
    def _grow(mat, a, s):
        m = len(a)
        out = np.empty((m + 1, m + 1))
        out[:m, :m] = mat + np.outer(a, a) / s
        out[:m, m] = -a / s
        out[m, :m] = -a / s
        out[m, m] = 1.0 / s
        return out

    def update(self, t: int, x: np.ndarray, y: float, w: float):
        if w == 0.0:
            return
        xr = x[None, :]
        kv = kern(self.kname, xr, self.d)[0]
        knn = float(kern(self.kname, xr, xr)[0, 0])
        a = self.kinv @ kv
        delta = knn - kv @ a
        if delta < 0.0:
            self.neg_delta += 1
        if delta > self.nu and len(self.d) < self.cap:
            self.kinv = self._grow(self.kinv, a, delta)
            self.p = self._grow(self.p, a, self.r * delta)
            self.alpha = np.append(self.alpha, 0.0)
            self.d = np.vstack([self.d, xr])
            kv = np.append(kv, knn)
            self.admitted.append(t)
            self.min_delta_adm = min(self.min_delta_adm, delta)
        elif self.atoms_only:
            return
        pk = self.p @ kv
        g = pk / (1.0 / w + kv @ pk)
        self.alpha = self.alpha + g * (y - kv @ self.alpha)
        self.p = self.p - np.outer(g, pk)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return kern(self.kname, x, self.d) @ self.alpha


class WindowKrls:
    """Sliding-window KRLS: every learned row of the last `width` rows is an
    atom; Q = (K_W + c diag(1/w))^-1 over them.  Add (Schur complement
    s = k(x, x) + c/w - kv' Q kv, qk = Q kv):
        Q <- [[Q + qk qk'/s, -qk/s], [-qk'/s, 1/s]]
    drop the oldest (e = Q_00, f = Q_1:,0):  Q <- Q_1:,1: - f f'/e.
    alpha = Q y_W, so the fit is KernelRidge(alpha=c, sample_weight=w) on the
    window rows by definition.  w == 0: the row only advances the clock (the
    window is counted in rows), and is not an atom."""

    def __init__(self, kname: str, k: int, width: int, c: float):
        self.kname, self.width, self.c = kname, width, c
        self.d = np.zeros((0, k))
        self.y = np.zeros(0)
        self.t = np.zeros(0, dtype=int)
        self.q = np.zeros((0, 0))

    def update(self, t: int, x: np.ndarray, y: float, w: float):
        while len(self.t) and self.t[0] <= t - self.width:
            e = self.q[0, 0]
            f = self.q[1:, 0]
            self.q = self.q[1:, 1:] - np.outer(f, f) / e
            self.d, self.y, self.t = self.d[1:], self.y[1:], self.t[1:]
        if w == 0.0:
            return
        xr = x[None, :]
        kv = kern(self.kname, xr, self.d)[0]
        knn = float(kern(self.kname, xr, xr)[0, 0])
        qk = self.q @ kv
        s = knn + self.c / w - kv @ qk
        self.q = AldKrls._grow(self.q, qk, s)
        self.d = np.vstack([self.d, xr])
        self.y = np.append(self.y, y)
        self.t = np.append(self.t, t)

    def alpha(self) -> np.ndarray:
        return self.q @ self.y

    def predict(self, x: np.ndarray) -> np.ndarray:
        return kern(self.kname, x, self.d) @ self.alpha()


def run_ald(kname, x, y, w, cap, nu, r=R_RIDGE, atoms_only=False):
    m = AldKrls(kname, x.shape[1], cap, nu, r, atoms_only)
    for t in range(len(x)):
        m.update(t, x[t], y[t], w[t])
    return m


def oracle_kr(kname, k, x, y, w, r=R_RIDGE):
    keep = w > 0
    return KernelRidge(alpha=r, **sk_kernel(kname, k)).fit(x[keep], y[keep], sample_weight=w[keep])


def oracle_sor(kname, k, dic, x, y, w, r=R_RIDGE):
    """Nystroem with every atom as a landmark (n_components = m, so its seeded
    permutation only reorders them), then Ridge(alpha=r, no intercept) on every
    learned row: min sum w (y - k(D, x)' alpha)^2 + r alpha' K_D alpha."""
    keep = w > 0
    ny = Nystroem(n_components=len(dic), random_state=0, **sk_kernel(kname, k)).fit(dic)
    rd = Ridge(alpha=r, fit_intercept=False, solver="cholesky")
    rd.fit(ny.transform(x[keep]), y[keep], sample_weight=w[keep])
    return lambda z: rd.predict(ny.transform(z))


def mse(a, b) -> float:
    return float(np.mean((a - b) ** 2))


def exp_krls_exact():
    """Where the online fit and a library fit are the same definition."""
    print("\n### Exact cases: the same definition, library oracle\n")
    print(
        "| kernel | k | case | rows learned | atoms | max abs pred gap (held-out) | "
        "max abs coef gap / max abs coef |"
    )
    print("|---|---|---|---|---|---|---|")
    worst = {}
    for kname in KERNELS:
        for k in KDIMS:
            for case in (
                "uncapped nu=0, first 150 rows",
                "atoms only, nu=1e-3, M=50",
                "nu=0, M=20, stop rule, all 2000 rows",
            ):
                pg = cg = 0.0
                n_at = 0
                n_rows = 0
                for seed in SEEDS:
                    x, y, xt, _ = make_regression(seed, N_ROWS, k)
                    w = np.ones(N_ROWS)
                    if case.startswith("uncapped"):
                        n_rows = 150
                        xs, ys, ws = x[:150], y[:150], w[:150]
                        m = run_ald(kname, xs, ys, ws, cap=10**9, nu=0.0)
                        ref = oracle_kr(kname, k, xs, ys, ws)
                        coef_ok = len(m.d) == 150
                    elif case.startswith("atoms"):
                        n_rows = N_ROWS
                        m = run_ald(kname, x, y, w, cap=50, nu=1e-3, atoms_only=True)
                        idx = np.array(m.admitted)
                        ref = oracle_kr(kname, k, x[idx], y[idx], w[idx])
                        coef_ok = True
                    else:
                        n_rows = N_ROWS
                        m = run_ald(kname, x, y, w, cap=20, nu=0.0)
                        f = oracle_sor(kname, k, m.d, x, y, w)
                        coef_ok = False
                    n_at = max(n_at, len(m.d))
                    pred = m.predict(xt)
                    ref_pred = f(xt) if case[0] == "n" else ref.predict(xt)
                    if coef_ok:
                        cg = max(
                            cg,
                            float(
                                np.max(np.abs(m.alpha - ref.dual_coef_))
                                / np.max(np.abs(ref.dual_coef_))
                            ),
                        )
                    pg = max(pg, float(np.max(np.abs(pred - ref_pred))))
                worst[(kname, case)] = max(worst.get((kname, case), 0.0), pg)
                cgs = fmt(cg) if case[0] != "n" else "n.a. (basis differs)"
                if case.startswith("uncapped") and n_at < 150:
                    cgs = f"n.a. (only {n_at} atoms: rank)"
                print(f"| {kname} | {k} | {case} | {n_rows} | {n_at} | {fmt(pg)} | {cgs} |")
    return worst


def exp_krls_grid():
    """nu = 1e-3, caps 20/50/200, N = 2000: growth, gaps, held-out error."""
    print(f"\n### Grid: nu = {NU}, r = {R_RIDGE}, N = {N_ROWS}, 5 seeds (worst gap, mean MSE)\n")
    print(
        "| kernel | k | M | atoms in rows 1-1000 / 1001-2000 (mean) | uncapped ALD atoms (mean) | "
        "pred gap: KRLS vs Nystroem+Ridge, same D, all rows | pred gap: atoms-only KRLS vs "
        "KernelRidge on D | pred gap: KRLS vs KernelRidge on D | held-out MSE: KRLS / KR on D / "
        "batch KR on all rows |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for kname in KERNELS:
        for k in KDIMS:
            unc = []
            for seed in SEEDS:
                x, y, _, _ = make_regression(seed, N_ROWS, k)
                unc.append(len(run_ald(kname, x, y, np.ones(N_ROWS), 10**9, NU).d))
            for cap in CAPS:
                g1 = g2 = g3 = 0.0
                a1 = a2 = 0.0
                e = np.zeros(3)
                for seed in SEEDS:
                    x, y, xt, yt = make_regression(seed, N_ROWS, k)
                    w = np.ones(N_ROWS)
                    m = run_ald(kname, x, y, w, cap, NU)
                    adm = np.array(m.admitted)
                    a1 += np.sum(adm < 1000) / len(SEEDS)
                    a2 += np.sum(adm >= 1000) / len(SEEDS)
                    pred = m.predict(xt)
                    sor = oracle_sor(kname, k, m.d, x, y, w)(xt)
                    mo = run_ald(kname, x, y, w, cap, NU, atoms_only=True)
                    kr_d = oracle_kr(kname, k, x[mo.admitted], y[mo.admitted], w[mo.admitted])
                    kr_pred = kr_d.predict(xt)
                    g1 = max(g1, float(np.max(np.abs(pred - sor))))
                    g2 = max(g2, float(np.max(np.abs(mo.predict(xt) - kr_pred))))
                    g3 = max(g3, float(np.max(np.abs(pred - kr_pred))))
                    batch = oracle_kr(kname, k, x, y, w).predict(xt)
                    e += np.array([mse(pred, yt), mse(kr_pred, yt), mse(batch, yt)]) / len(SEEDS)
                rows.append((kname, k, cap, a1, a2, g1, g2, g3, e))
                print(
                    f"| {kname} | {k} | {cap} | {a1:.1f} / {a2:.1f} | {np.mean(unc):.1f} | "
                    f"{fmt(g1)} | {fmt(g2)} | {fmt(g3)} | "
                    f"{e[0]:.4f} / {e[1]:.4f} / {e[2]:.4f} |"
                )
    print("\n(noise variance 0.0100: the floor for every held-out MSE)")
    return rows


def exp_krls_window():
    """Sliding window of the last W rows against KernelRidge on those rows."""
    print("\n### Window: last W rows, c = r = 0.1, against KernelRidge on the window rows\n")
    print(
        "| kernel | k | W | max abs pred gap (held-out, every 100 rows) | "
        "max coef gap / max coef | held-out MSE at the end |"
    )
    print("|---|---|---|---|---|---|")
    for kname in KERNELS:
        for k in KDIMS:
            for width in CAPS:
                pg = cg = 0.0
                em = 0.0
                for seed in SEEDS:
                    x, y, xt, yt = make_regression(seed, N_ROWS, k)
                    w = np.ones(N_ROWS)
                    m = WindowKrls(kname, k, width, R_RIDGE)
                    for t in range(N_ROWS):
                        m.update(t, x[t], y[t], w[t])
                        if (t + 1) % 100 == 0 and t + 1 >= width:
                            lo = t + 1 - width
                            ref = oracle_kr(kname, k, x[lo : t + 1], y[lo : t + 1], w[lo : t + 1])
                            pg = max(pg, float(np.max(np.abs(m.predict(xt) - ref.predict(xt)))))
                            al = m.alpha()
                            cg = max(
                                cg,
                                float(
                                    np.max(np.abs(al - ref.dual_coef_))
                                    / np.max(np.abs(ref.dual_coef_))
                                ),
                            )
                    em += mse(m.predict(xt), yt) / len(SEEDS)
                print(f"| {kname} | {k} | {width} | {fmt(pg)} | {fmt(cg)} | {em:.4f} |")


def exp_krls_zero_weight():
    print("\n### Zero-weight rows (first 50 rows, then one row in ten, at w = 0)\n")
    k = 2
    x, y, xt, _ = make_regression(3, N_ROWS, k)
    rng = np.random.default_rng(77)
    w = rng.uniform(0.5, 1.5, N_ROWS) * (rng.random(N_ROWS) >= 0.1)
    w[:50] = 0.0
    for kname in KERNELS:
        ald = AldKrls(kname, k, 50, NU, R_RIDGE)
        ald0 = AldKrls(kname, k, 20, 0.0, R_RIDGE)
        win = WindowKrls(kname, k, 200, R_RIDGE)
        unchanged = True
        nan = False
        for t in range(N_ROWS):
            before = [(o.d.copy(), o.alpha.copy(), o.p.copy(), o.kinv.copy()) for o in (ald, ald0)]
            wb = (win.d.copy(), win.q.copy(), win.y.copy())
            for o in (ald, ald0, win):
                o.update(t, x[t], y[t], w[t])
            if w[t] == 0.0:
                for o, b in zip((ald, ald0), before, strict=True):
                    unchanged &= all(
                        np.array_equal(u, v)
                        for u, v in zip(b, (o.d, o.alpha, o.p, o.kinv), strict=True)
                    )
                # the window may only have dropped its oldest atom (the clock)
                dropped = len(wb[2]) - len(win.y)
                unchanged &= dropped in (0, 1) and np.array_equal(wb[2][dropped:], win.y)
            nan |= not all(np.isfinite(a).all() for a in (ald.alpha, ald.p, ald0.p, win.q))
        idx = np.array(ald0.admitted)
        f = oracle_sor(kname, k, ald0.d, x, y, w)
        lo = N_ROWS - 200
        kr_w = oracle_kr(kname, k, x[lo:], y[lo:], w[lo:])
        g_sor = float(np.max(np.abs(ald0.predict(xt) - f(xt))))
        g_win = float(np.max(np.abs(win.predict(xt) - kr_w.predict(xt))))
        print(
            f"- {kname}: NaN in any state: {nan}; zero-weight rows leave the ALD states "
            f"bit-identical and the window's only by its clock: {unchanged}; first atom at row "
            f"{idx[0]}; nu = 0, M = 20 vs Nystroem+Ridge(sample_weight): {fmt(g_sor)}; window "
            f"W = 200 vs KernelRidge(sample_weight): {fmt(g_win)}"
        )


def exp_krls_numerics():
    """Positive definiteness of Kinv and P against nu, near the cap."""
    print("\n### Numerical risk: Kinv and P against nu (k = 2, M = 200, N = 2000, seed 0)\n")
    print(
        "| kernel | nu | atoms | smallest admitted delta | delta < 0 seen | "
        "min eig sym(Kinv) | rel err Kinv vs inv(K_D) | min eig sym(P) | cond K_D | "
        "pred gap vs Nystroem+Ridge |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    k = 2
    x, y, xt, _ = make_regression(0, N_ROWS, k)
    w = np.ones(N_ROWS)
    for kname in KERNELS:
        for nu in (1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 0.0):
            with np.errstate(all="ignore"):
                m = run_ald(kname, x, y, w, 200, nu)
                kd = kern(kname, m.d, m.d)
                ki = 0.5 * (m.kinv + m.kinv.T)
                ev_k = float(np.linalg.eigvalsh(ki)[0])
                ev_p = float(np.linalg.eigvalsh(0.5 * (m.p + m.p.T))[0])
                try:
                    ref = np.linalg.inv(kd)
                    rel = float(np.linalg.norm(m.kinv - ref) / np.linalg.norm(ref))
                except np.linalg.LinAlgError:
                    rel = float("nan")
                cond = float(np.linalg.cond(kd))
                gap = float(np.max(np.abs(m.predict(xt) - oracle_sor(kname, k, m.d, x, y, w)(xt))))
            print(
                f"| {kname} | {nu:g} | {len(m.d)} | {m.min_delta_adm:.1e} | {m.neg_delta} | "
                f"{ev_k:.1e} | {rel:.1e} | {ev_p:.1e} | {cond:.1e} | {fmt(gap)} |"
            )


def exp_krls_cost():
    print("\n### State bytes and per-row cost\n")
    print(
        "| M | k | ALD-KRLS: D + Kinv + P + alpha, dense | packed (two triangles) | "
        "window: rows + y + t + Q, dense | flops per row, ALD (about 8 M^2) | "
        "measured us/row (ALD, cap full) | measured us/row (window, full) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for cap in CAPS:
        for k in KDIMS:
            dense = 8 * (cap * k + 2 * cap * cap + cap)
            packed = 8 * (cap * k + cap * (cap + 1) + cap)
            win = 8 * (cap * k + cap + cap + cap * cap)
            x, y, _, _ = make_regression(0, 3 * cap + 400, k)
            m = AldKrls("rbf", k, cap, -1.0, R_RIDGE)  # nu < 0: admit until full
            wm = WindowKrls("rbf", k, cap, R_RIDGE)
            for t in range(cap):
                m.update(t, x[t], y[t], 1.0)
                wm.update(t, x[t], y[t], 1.0)
            t0 = time.perf_counter()
            for t in range(cap, cap + 400):
                m.update(t, x[t], y[t], 1.0)
            t1 = time.perf_counter()
            for t in range(cap, cap + 400):
                wm.update(t, x[t], y[t], 1.0)
            t2 = time.perf_counter()
            print(
                f"| {cap} | {k} | {dense:,} | {packed:,} | {win:,} | {8 * cap * cap:,} | "
                f"{(t1 - t0) / 400 * 1e6:.0f} | {(t2 - t1) / 400 * 1e6:.0f} |"
            )


def main_krls(parts):
    t0 = time.time()
    parts = parts or ["exact", "grid", "window", "zero", "numerics", "cost"]
    if "exact" in parts:
        exp_krls_exact()
    if "grid" in parts:
        exp_krls_grid()
    if "window" in parts:
        exp_krls_window()
    if "zero" in parts:
        exp_krls_zero_weight()
    if "numerics" in parts:
        exp_krls_numerics()
    if "cost" in parts:
        exp_krls_cost()
    print(f"\n(total {time.time() - t0:.1f} s)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "pca"
    if which == "pca":
        main_pca()
    elif which == "krls":
        main_krls(sys.argv[2:])
    else:
        raise SystemExit(f"unknown part {which!r}; parts: 'pca', 'krls'")
