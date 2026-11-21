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

Run: uv run python scripts/study_505_502.py pca   (about 20 s)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.linalg as sla
from sklearn.decomposition import PCA, IncrementalPCA

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


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "pca"
    if which == "pca":
        main_pca()
    else:
        raise SystemExit(f"unknown part {which!r}; part 1 has 'pca'")
