"""Online dependence measures for study 510: binned mutual information and
windowed rank correlations, one row at a time.

Every pair is a feature ``j`` (of ``k``) against a target ``t`` (of ``T``), as
``marginal`` pairs them; never feature against feature.

Bins (``marginal``'s, ``margbins.rs``)
--------------------------------------
A value ``v`` falls in bin ``edges.partition_point(e <= v)``, numpy's
``searchsorted(edges, v, side="right")``: a value equal to an edge goes to the
bin above it, and a non-finite value goes in no bin. ``edges_from`` is
``margbins::edges_from`` line for line: ``"fixed"`` cuts ``[lo, hi]`` of a
warm-up sample into equal widths, ``"quantile"`` closes each bin at the weight
still unassigned over the bins still to come (so a point mass fills one bin);
both drop repeated edges and any edge at or below the sample's minimum.

Mutual information (``BinnedMI``)
---------------------------------
Each pair keeps a joint table of weights ``n_ab`` over (feature bin ``a``,
target bin ``b``). A row of weight ``w > 0`` whose feature and target both fall
in a bin adds ``w`` to one cell of that pair's table. With ``N = sum n_ab`` and
the margins ``n_a = sum_b n_ab``, ``n_b = sum_a n_ab``, the plug-in estimate is
read from the table, ``ln`` at read time only::

    MI = sum_{n_ab > 0} (n_ab / N) * (ln n_ab + ln N - ln n_a - ln n_b)      (nats)

The Miller-Madow correction applies ``H_MM = H + (m - 1) / (2N)`` (``m`` the
nonzero cells) to each of ``H(X)``, ``H(Y)``, ``H(X, Y)``, and
``MI = H(X) + H(Y) - H(X, Y)`` gives::

    MI_MM = MI - (B_xy - B_x - B_y + 1) / (2N)

with ``B_xy``, ``B_x``, ``B_y`` the nonzero joint cells and margins: the
plug-in is biased upward by about ``(b_x - 1)(b_y - 1) / (2N)`` at
independence, and the correction subtracts it. It is given only for unit
weights (a window or no decay): under decay the ``N`` it needs is not the
table's weight, and no library settles which.

Decay (half-life ``h`` rows, ``lam = 2^(-1/h)``): ``n_ab(t) = sum_i w_i
lam^(t - t_i)`` over the rows in the cell. As in ``margbins.rs`` the table is
kept undecayed against one scale ``s_t = prod lam``: a row adds ``w / s_t``,
nothing else moves, and when ``s`` falls below ``1e-50`` it is folded into the
table and reset to 1. MI is scale-free, so a read needs no fold.

Window (the last ``W`` rows of the stream): a ring of the rows' bin indices
(``uint16``, ``NO_BIN`` for a value in no bin) and weights. The row leaving
subtracts its weight from the cells it added to; unit weights keep every cell
an integer, so the subtraction is exact. A zero-weight row and a row with a
non-finite value take a slot (they advance the clock) and add to no cell.

Ranks (``RankWindow``)
----------------------
A ring of the last ``W`` rows' values (``k + T`` doubles) and weights. Pair
``(j, t)``'s sample is the window's rows with weight ``> 0`` and both values
finite. Weights only gate: a rank measure counts each row once.

Kendall's tau-b is kept *incrementally*: for each pair, ``S = n_c - n_d``, the
count ``n_0`` of pairs of rows, and ``n_1`` / ``n_2`` the pairs of rows tied in ``x`` / in
``y``. A row entering compares itself with the rows in the ring (``O(W)``
per pair), adding ``sign(dx) sign(dy)`` to ``S`` and its ties to ``n_1`` and
``n_2``; the row leaving subtracts the same against the rows that stay. Every
counter is an integer, so the read is exact::

    tau_b = S / sqrt((n_0 - n_1)(n_0 - n_2))      (n_0 = n (n - 1) / 2 for n rows)

Spearman's rho and Chatterjee's xi are read every ``rank_every`` rows from the
ring (``O(W log W)`` per pair) and held between reads. Spearman: the Pearson
correlation of the average ranks (``(lo + hi + 1) / 2`` from the two
``searchsorted`` sides of the sorted sample). Chatterjee (2021, JASA 116,
eq. 1.1 and 1.2): sort the pairs by ``x``, let ``r_i = #{j: y_j <= y_(i)}``
and ``l_i = #{j: y_j >= y_(i)}``; then::

    xi_n = 1 - n sum |r_{i+1} - r_i| / (2 sum l_i (n - l_i))        (ties in y)
    xi_n = 1 - 3 sum |r_{i+1} - r_i| / (n^2 - 1)                       (no ties)

The paper breaks ties in ``x`` at random; this study breaks them by arrival
(a stable sort of the window oldest row first), which is deterministic, so
chunk invariance holds, and needs no random stream in the state.

What a row emits (``RankWindow.step``) is read *before* the row is learned, so a
row's numbers never include it (rule 2's analogue). Kendall is read every row,
Spearman and xi on rows ``0, rank_every, 2 rank_every, ...`` of the stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NO_BIN = np.iinfo(np.uint16).max
RENORM_AT = 1e-50


def edges_from(rule: str, n_bins: int, values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """``margbins::edges_from``: strictly increasing finite edges from a warm-up sample."""
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]
    if len(v) < 2 or n_bins < 2:
        return np.empty(0)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    n = len(v)
    lo, hi = v[0], v[-1]
    edges: list[float] = []
    if rule == "quantile":
        cum = np.cumsum(w)
        total = cum[-1]
        start = 0
        for remaining in range(n_bins, 1, -1):
            base = 0.0 if start == 0 else cum[start - 1]
            share = (total - base) / remaining
            if not np.isfinite(share) or share <= 0.0:
                break
            i = start + int(np.searchsorted(cum[start:] - base, share, side="right"))
            if i >= n:
                break
            q = v[i]
            if q > v[start]:
                edge = q
            else:
                above = v[i:][v[i:] > q]
                if len(above) == 0:
                    break
                edge = above[0]
            edges.append(float(edge))
            start += int(np.searchsorted(v[start:], edge, side="left"))
    elif rule == "fixed":
        width = (hi - lo) / n_bins
        edges = [lo + width * kk for kk in range(1, n_bins)]
    else:
        raise ValueError(rule)
    out: list[float] = []
    for e in edges:
        if np.isfinite(e) and e > lo and (not out or e != out[-1]):
            out.append(e)
    return np.asarray(out)


def bins_of(edges: list[np.ndarray], row: np.ndarray) -> np.ndarray:
    """Each value's bin (``uint16``), ``NO_BIN`` where it is not finite."""
    out = np.full(len(edges), NO_BIN, dtype=np.uint16)
    for j, (e, v) in enumerate(zip(edges, row, strict=True)):
        if np.isfinite(v):
            out[j] = np.searchsorted(e, v, side="right")
    return out


def mi_from_tables(c: np.ndarray) -> dict[str, np.ndarray]:
    """Plug-in MI, Miller-Madow MI and the counts it uses, per table over the last two axes."""
    n = c.sum(axis=(-2, -1))
    na = c.sum(axis=-1)
    nb = c.sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln = np.where(c > 0, np.log(np.where(c > 0, c, 1.0)), 0.0)
        lna = np.log(np.where(na > 0, na, 1.0))
        lnb = np.log(np.where(nb > 0, nb, 1.0))
        lnn = np.log(np.where(n > 0, n, 1.0))
        term = ln + lnn[..., None, None] - lna[..., :, None] - lnb[..., None, :]
        mi = np.where(c > 0, c * term, 0.0).sum(axis=(-2, -1)) / np.where(n > 0, n, np.nan)
    bxy = (c > 0).sum(axis=(-2, -1))
    bx = (na > 0).sum(axis=-1)
    by = (nb > 0).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mm = mi - (bxy - bx - by + 1) / (2.0 * n)
    return {"mi": mi, "mi_mm": mm, "n": n, "b_xy": bxy, "b_x": bx, "b_y": by}


@dataclass
class BinnedMI:
    """Joint bin counts per (feature, target) pair; see the module doc."""

    x_edges: list[np.ndarray]
    y_edges: list[np.ndarray]
    half_life: float | None = None
    window: int | None = None

    def __post_init__(self) -> None:
        if self.half_life is not None and self.window is not None:
            raise ValueError("decay or a window, not both, in this study")
        self.k, self.t = len(self.x_edges), len(self.y_edges)
        self.bx = max(len(e) for e in self.x_edges) + 1
        self.by = max(len(e) for e in self.y_edges) + 1
        if max(self.bx, self.by) >= NO_BIN:
            raise ValueError("too many bins for uint16")
        self.counts = np.zeros((self.k, self.t, self.bx, self.by))
        self.lam = 1.0 if self.half_life is None else 0.5 ** (1.0 / self.half_life)
        self.scale = 1.0
        self.rows = 0
        if self.window is not None:
            self.ring_bins = np.full((self.window, self.k + self.t), NO_BIN, dtype=np.uint16)
            self.ring_w = np.zeros(self.window)
            self.head = 0

    def _add(self, b: np.ndarray, u: float) -> None:
        bx, by = b[: self.k], b[self.k :]
        js = np.flatnonzero(bx != NO_BIN)
        ts = np.flatnonzero(by != NO_BIN)
        if len(js) and len(ts):
            jj, tt = np.meshgrid(js, ts, indexing="ij")
            self.counts[jj, tt, bx[jj], by[tt]] += u

    def update(self, x: np.ndarray, y: np.ndarray, w: float) -> None:
        if not (np.isfinite(w) and w >= 0.0):
            raise ValueError("weight must be finite and >= 0")
        if self.rows > 0 and self.lam != 1.0:
            self.scale *= self.lam
            if self.scale < RENORM_AT:
                self.counts *= self.scale
                self.scale = 1.0
        b = np.concatenate([bins_of(self.x_edges, x), bins_of(self.y_edges, y)])
        if self.window is not None:
            if self.rows >= self.window and self.ring_w[self.head] > 0.0:
                self._add(self.ring_bins[self.head], -self.ring_w[self.head])
            self.ring_bins[self.head] = b
            self.ring_w[self.head] = w
            self.head = (self.head + 1) % self.window
        if w > 0.0:
            self._add(b, w / self.scale)
        self.rows += 1

    def tables(self) -> np.ndarray:
        """The decayed weights, ``(k, T, bx, by)``."""
        return self.counts * self.scale

    def read(self) -> dict[str, np.ndarray]:
        out = mi_from_tables(self.counts)
        out["n"] = out["n"] * self.scale
        if self.half_life is not None:
            out["mi_mm"] = np.full_like(out["mi"], np.nan)
        return out

    def state_arrays(self) -> dict[str, np.ndarray]:
        """Every array the state holds (edges, table, ring, scalars)."""
        arr = {"counts": self.counts, "scalars": np.array([self.lam, self.scale, float(self.rows)])}
        for j, e in enumerate(self.x_edges):
            arr[f"x_edges{j}"] = e
        for t, e in enumerate(self.y_edges):
            arr[f"y_edges{t}"] = e
        if self.window is not None:
            arr["ring_bins"] = self.ring_bins
            arr["ring_w"] = self.ring_w
            arr["head"] = np.array([self.head], dtype=np.int64)
        return arr


def avg_ranks(v: np.ndarray) -> np.ndarray:
    s = np.sort(v)
    lo = np.searchsorted(s, v, side="left")
    hi = np.searchsorted(s, v, side="right")
    return (lo + hi + 1) / 2.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return np.nan
    a = avg_ranks(x) - (n + 1) / 2.0
    b = avg_ranks(y) - (n + 1) / 2.0
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / den) if den > 0 else np.nan


def chatterjee_xi(x: np.ndarray, y: np.ndarray, y_continuous: bool = False) -> float:
    n = len(x)
    if n < 2:
        return np.nan
    yo = y[np.argsort(x, kind="stable")]
    s = np.sort(y)
    r = np.searchsorted(s, yo, side="right").astype(float)
    num = np.abs(np.diff(r)).sum()
    if y_continuous:
        return float(1.0 - 3.0 * num / (n * n - 1.0))
    l_ = (n - np.searchsorted(s, yo, side="left")).astype(float)
    den = 2.0 * np.sum((n - l_) * l_)
    return float(1.0 - n * num / den) if den > 0 else np.nan


class RankWindow:
    """Windowed Spearman, Kendall tau-b and Chatterjee xi; see the module doc."""

    def __init__(self, k: int, t: int, window: int, rank_every: int = 1) -> None:
        self.k, self.t, self.window, self.rank_every = k, t, window, rank_every
        self.xr = np.full((window, k), np.nan)
        self.yr = np.full((window, t), np.nan)
        self.wr = np.zeros(window)
        self.head = 0
        self.rows = 0
        z = np.zeros((k, t), dtype=np.int64)
        self.s, self.n, self.n1, self.n2 = z.copy(), z.copy(), z.copy(), z.copy()
        self.held = np.full((2, k, t), np.nan)  # spearman, xi

    def _pairs(self, x: np.ndarray, y: np.ndarray, w: float, skip: int, sign: int) -> None:
        """Add (``sign`` 1) or remove (-1) row ``(x, y, w)``'s pairs with every ring row but ``skip``."""
        if not w > 0.0:
            return
        fx, fy = np.isfinite(x), np.isfinite(y)
        if not (fx.any() and fy.any()):
            return
        live = self.wr > 0.0
        live[skip] = False
        vx = np.isfinite(self.xr) & live[:, None] & fx[None, :]
        vy = np.isfinite(self.yr) & live[:, None] & fy[None, :]
        with np.errstate(invalid="ignore"):
            dx = np.where(vx, x[None, :] - self.xr, 0.0)
            dy = np.where(vy, y[None, :] - self.yr, 0.0)
        sx, sy = np.sign(dx), np.sign(dy)
        tx = (vx & (dx == 0.0)).astype(float)
        ty = (vy & (dy == 0.0)).astype(float)
        fvx, fvy = vx.astype(float), vy.astype(float)
        self.s += sign * np.rint(sx.T @ sy).astype(np.int64)
        self.n += sign * np.rint(fvx.T @ fvy).astype(np.int64)
        self.n1 += sign * np.rint(tx.T @ fvy).astype(np.int64)
        self.n2 += sign * np.rint(fvx.T @ ty).astype(np.int64)

    def kendall(self) -> np.ndarray:
        n0 = self.n  # the pair count itself
        den = np.sqrt((n0 - self.n1).astype(float) * (n0 - self.n2).astype(float))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, self.s / np.where(den > 0, den, 1.0), np.nan)

    def sample(self, j: int, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Pair ``(j, t)``'s sample from the ring, oldest row first."""
        order = (self.head + np.arange(self.window)) % self.window
        xs, ys, ws = self.xr[order, j], self.yr[order, t], self.wr[order]
        m = (ws > 0.0) & np.isfinite(xs) & np.isfinite(ys)
        return xs[m], ys[m]

    def read_ranks(self) -> np.ndarray:
        out = np.full((2, self.k, self.t), np.nan)
        order = (self.head + np.arange(self.window)) % self.window  # oldest first
        xr, yr, live = self.xr[order], self.yr[order], self.wr[order] > 0.0
        for j in range(self.k):
            for t in range(self.t):
                m = live & np.isfinite(xr[:, j]) & np.isfinite(yr[:, t])
                xs, ys = xr[m, j], yr[m, t]
                out[0, j, t] = spearman(xs, ys)
                out[1, j, t] = chatterjee_xi(xs, ys)
        return out

    def step(self, x: np.ndarray, y: np.ndarray, w: float) -> dict[str, np.ndarray]:
        """Emit the numbers before this row, then learn it."""
        if self.rows % self.rank_every == 0:
            self.held = self.read_ranks()
        out = {"spearman": self.held[0].copy(), "kendall": self.kendall(), "xi": self.held[1].copy()}
        self.learn(x, y, w)
        return out

    def learn(self, x: np.ndarray, y: np.ndarray, w: float) -> None:
        if not (np.isfinite(w) and w >= 0.0):
            raise ValueError("weight must be finite and >= 0")
        h = self.head
        if self.rows >= self.window:
            self._pairs(self.xr[h].copy(), self.yr[h].copy(), self.wr[h], h, -1)
        self.xr[h], self.yr[h], self.wr[h] = x, y, w
        self._pairs(x, y, w, h, 1)
        self.head = (h + 1) % self.window
        self.rows += 1

    def state_arrays(self) -> dict[str, np.ndarray]:
        return {
            "xr": self.xr, "yr": self.yr, "wr": self.wr,
            "s": self.s, "n": self.n, "n1": self.n1, "n2": self.n2,
            "held": self.held,
            "scalars": np.array([self.head, self.rows], dtype=np.int64),
        }


def run_ranks(rw: RankWindow, x: np.ndarray, y: np.ndarray, w: np.ndarray, chunks: int = 1) -> dict[str, np.ndarray]:
    """Feed the stream in ``chunks`` pieces; the emitted numbers, ``(n, k, T)`` each."""
    outs: dict[str, list[np.ndarray]] = {"spearman": [], "kendall": [], "xi": []}
    for piece in np.array_split(np.arange(len(w)), chunks):
        for i in piece:
            o = rw.step(x[i], y[i], float(w[i]))
            for key in outs:
                outs[key].append(o[key])
    return {key: np.array(v) for key, v in outs.items()}
