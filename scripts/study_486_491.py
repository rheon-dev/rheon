# ruff: noqa: E501  (the printed Markdown table headers are one line each)
"""Study 486-491: online AR(p) and spectral estimates against statsmodels and scipy.

Part 1 (task 486): AR(p) two ways, each with its lags kept in a ring of the last
``p`` learned values.

* ``OnlineARLS``: exponentially weighted least squares on z_t = (1, y_{t-1}, ...,
  y_{t-p}); the decayed Gram G and cross term b, solved by Cholesky each row.
* ``OnlineARYW``: Yule-Walker from decayed running lag sums, solved at emit by the
  Levinson-Durbin recursion.

Oracles: statsmodels ``AutoReg(trend='c').fit().params`` (OLS) and ``yule_walker``
with no decay; under decay, batch weighted least squares written longhand
(``numpy.linalg.lstsq`` on the rows, with ``statsmodels`` ``WLS`` as a second
opinion) and the weighted autocovariances written longhand fed to statsmodels'
``levinson_durbin(isacov=True)``.

Every series is generated here from a seed; nothing is read from or written to
disk. Run: ``.venv/bin/python scripts/study_486_491.py [ar]``. Parts 2 (spectral,
491) and 3 (order selection) add their sections to this file.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.linalg
import scipy.signal
from statsmodels.regression.linear_model import WLS, yule_walker
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import levinson_durbin

SEEDS = (0, 1, 2, 3, 4)
ORDERS = (1, 5, 20)
LENGTHS = (2000, 20000)


# --------------------------------------------------------------------------- data


def ar_coefficients(p: int, rng: np.random.Generator) -> np.ndarray:
    """phi_1..phi_p of a stationary AR(p): roots of z^p - phi_1 z^{p-1} - ... - phi_p
    drawn inside the unit circle (moduli 0.3 to 0.85, conjugate pairs)."""
    roots: list[complex] = []
    while len(roots) + 2 <= p:
        r, a = rng.uniform(0.3, 0.85), rng.uniform(0.1, np.pi - 0.1)
        roots += [r * np.exp(1j * a), r * np.exp(-1j * a)]
    if len(roots) < p:
        roots.append(complex(rng.uniform(0.3, 0.85) * rng.choice([-1.0, 1.0])))
    phi = -np.real(np.poly(roots))[1:]
    # check: the reciprocal roots of 1 - sum phi_i L^i are the roots above
    assert np.max(np.abs(np.roots(np.r_[1.0, -phi]))) < 0.86
    return phi


def ar_series(p: int, n: int, seed: int, c: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phi = ar_coefficients(p, rng)
    burn = 1000
    e = rng.standard_normal(n + burn)
    y = scipy.signal.lfilter([1.0], np.r_[1.0, -phi], c + e)[burn:]
    return y, phi


# --------------------------------------------------------------------------- models


class LagRing:
    """The last p learned values. Advances only on a learned row (weight > 0):
    a zero-weight or missing row is not a lag ("a lag counts learned rows")."""

    def __init__(self, p: int):
        self.p, self.buf, self.pos, self.count = p, np.zeros(p), 0, 0

    def lags(self) -> np.ndarray:
        """(y_{t-1}, ..., y_{t-p}) for the rows held; the first min(count, p) are valid."""
        return self.buf[(self.pos - 1 - np.arange(self.p)) % self.p]

    def push(self, y: float) -> None:
        self.buf[self.pos] = y
        self.pos = (self.pos + 1) % self.p
        self.count += 1

    nbytes = property(lambda self: self.buf.nbytes + 16)  # values + pos + count


class OnlineARLS:
    """AR(p) by exponentially weighted least squares, sum form.

    pred_t = z_t' beta_{t-1}                       (out of sample: before the update)
    G_t = lam G_{t-1} + w_t z_t z_t'               b_t = lam b_{t-1} + w_t z_t y_t
    W_t = lam W_{t-1} + w_t                        (n_eff before the row: W_{t-1})
    beta_t = G_t^{-1} b_t by Cholesky, when at least k = p + 1 rows were learned
    and the factorization succeeds; otherwise beta stays undefined (NaN).

    A row is learned only once the ring holds p values (AutoReg's hold_back = p);
    the earlier rows fill the ring. A zero-weight row decays the state and adds
    nothing (no 0 * y product is formed, so a NaN target cannot leak in); beta is
    unchanged because G and b scale together, so no solve is needed on it.
    """

    def __init__(self, p: int, lam: float = 1.0, solve_every_row: bool = True, shift: bool = True):
        k = p + 1
        self.p, self.k, self.lam, self.every = p, k, lam, solve_every_row
        self.shift, self.origin = shift, None
        self.ring = LagRing(p)
        self.G, self.b = np.zeros((k, k)), np.zeros(k)
        self.W, self.learned = 0.0, 0
        self.beta = np.full(k, np.nan)

    def update(self, y: float, w: float) -> float:
        if w > 0.0 and self.origin is None:
            self.origin = y if self.shift else 0.0
        y = y - self.origin if self.origin is not None else y
        z = np.empty(self.k)
        z[0] = 1.0
        z[1:] = self.ring.lags()
        full = self.ring.count >= self.p
        pred = float(z @ self.beta) + self.origin if full else np.nan
        if self.lam != 1.0:
            self.G *= self.lam
            self.b *= self.lam
            self.W *= self.lam
        if w > 0.0:
            if full:
                self.G += w * np.outer(z, z)
                self.b += (w * y) * z
                self.W += w
                self.learned += 1
                if self.every:
                    self.solve()
            self.ring.push(y)
        return pred

    def solve(self) -> np.ndarray:
        if self.learned >= self.k:
            try:
                self.beta = scipy.linalg.cho_solve(scipy.linalg.cho_factor(self.G), self.b)
            except np.linalg.LinAlgError:
                self.beta = np.full(self.k, np.nan)
        return self.coefficients()

    def coefficients(self) -> np.ndarray:
        """(c, phi_1..phi_p) on the original scale: y - o = c' + sum phi_i (y_{t-i} - o)
        gives c = c' + o (1 - sum phi)."""
        out = self.beta.copy()
        if self.origin is not None:
            out[0] += self.origin * (1.0 - out[1:].sum())
        return out

    @property
    def nbytes(self) -> int:
        return (
            self.ring.nbytes + self.G.nbytes + self.b.nbytes + self.beta.nbytes + 24
        )  # W, learned, origin


def levinson(gamma: np.ndarray, p: int) -> tuple[np.ndarray, float]:
    """Levinson-Durbin on gamma_0..gamma_p: phi (order p) and the innovation
    variance. O(p^2). Stops with NaN if a prediction error variance is not > 0."""
    phi = np.zeros(p)
    v = gamma[0]
    if not v > 0.0:
        return np.full(p, np.nan), np.nan
    for m in range(1, p + 1):
        kappa = (gamma[m] - phi[: m - 1] @ gamma[m - 1 : 0 : -1]) / v
        prev = phi[: m - 1].copy()
        phi[: m - 1] = prev - kappa * prev[::-1]
        phi[m - 1] = kappa
        v *= 1.0 - kappa * kappa
        if not v > 0.0:
            return np.full(p, np.nan), np.nan
    return phi, v


class OnlineARYW:
    """AR(p) by Yule-Walker on decayed running lag sums, in the tapered form.

    Each learned row s carries a weight W_s = w_s lam^(age of s on the clock)
    and an amplitude a_s = sqrt(W_s). The autocovariances are those of the
    tapered, demeaned sequence u_s = a_s (y_s - m), m = sum W_s y_s / sum W_s:
        S_k = sum_{s>=k} u_s u_{s-k}
            = C_k - m (Y_k + L_k) + m^2 P_k,   over the pairs of learned rows, with
        P_k = sum a_s a_{s-k}    Y_k = sum a_s a_{s-k} y_s
        L_k = sum a_s a_{s-k} y_{s-k}    C_k = sum a_s a_{s-k} y_s y_{s-k}
        gamma_k = S_k / P_0   ('mle')   or   S_k / P_k   ('adjusted'),
    then Levinson-Durbin. P_0 = sum W_s and Y_0 = L_0 = sum W_s y_s. The ring keeps
    each lag's value and its current amplitude; every row multiplies the
    amplitudes by sqrt(lam) and the sums by lam, so a pair's weight
    a_s a_{s-k} = sqrt(W_s W_{s-k}) stays exact across zero-weight rows.

    'mle' is positive definite by construction (the autocovariance of a finite
    sequence), so the recursion's prediction variance stays > 0. At lam = 1 and
    unit weights this is exactly statsmodels' acovf with demean=True (the
    full-sample mean): Y_k is the sum without the first k values, L_k the sum
    without the last k. (A form that gives a pair the later row's weight W_s
    alone is not positive definite under decay; the study found it failing.)
    """

    def __init__(self, p: int, lam: float = 1.0, shift: bool = True):
        self.p, self.lam, self.root_lam = p, lam, np.sqrt(lam)
        self.shift, self.origin = shift, None
        self.ring = LagRing(p)
        self.amp = np.zeros(p)  # amplitudes, stored at the ring's positions
        self.sums = np.zeros((4, p + 1))  # P, Y, L, C

    def update(self, y: float, w: float) -> None:
        if self.lam != 1.0:
            self.sums *= self.lam
            self.amp *= self.root_lam
        if w > 0.0:
            if self.origin is None:
                self.origin = y if self.shift else 0.0
            y = y - self.origin
            kk = min(self.ring.count, self.p)
            order = (self.ring.pos - 1 - np.arange(self.p)) % self.p
            lags = self.ring.buf[order][:kk]
            a = np.sqrt(w)
            pw = np.empty(kk + 1)
            pw[0] = w
            pw[1:] = a * self.amp[order][:kk]
            P, Y, L, C = self.sums
            P[: kk + 1] += pw
            Y[: kk + 1] += pw * y
            L[0] += w * y
            L[1 : kk + 1] += pw[1:] * lags
            C[0] += w * y * y
            C[1 : kk + 1] += (pw[1:] * y) * lags
            self.amp[self.ring.pos] = a
            self.ring.push(y)

    def gamma(self, method: str = "mle") -> np.ndarray:
        P, Y, L, C = self.sums
        if not P[0] > 0.0:
            return np.full(self.p + 1, np.nan)
        m = Y[0] / P[0]
        S = C - m * (Y + L) + m * m * P
        if method == "mle":
            return S / P[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(P > 0.0, S / np.where(P > 0.0, P, 1.0), np.nan)

    def solve(self, method: str = "mle") -> tuple[np.ndarray, float]:
        return levinson(self.gamma(method), self.p)

    @property
    def nbytes(self) -> int:
        return self.ring.nbytes + self.amp.nbytes + self.sums.nbytes + 8  # origin


# --------------------------------------------------------------------------- oracles


def lag_design(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows t = p..n-1 of (1, y_{t-1}, ..., y_{t-p}) and y_t."""
    n = len(y)
    X = np.column_stack([np.ones(n - p)] + [y[p - i : n - i] for i in range(1, p + 1)])
    return X, y[p:]


def wls_longhand(y_learned, ages, w_learned, p, lam):
    """Batch WLS on the learned rows: row weight w_s lam^(age_s)."""
    X, t = lag_design(y_learned, p)
    W = w_learned[p:] * lam ** ages[p:]
    sw = np.sqrt(W)
    beta = np.linalg.lstsq(X * sw[:, None], t * sw, rcond=None)[0]
    beta_sm = WLS(t, X, weights=W).fit().params
    return beta, beta_sm


def yw_longhand(y_learned, ages, w_learned, p, lam, method):
    """The tapered weighted autocovariances written out over the rows; statsmodels'
    levinson_durbin does the algebra. Also, for 'mle', statsmodels' yule_walker on
    the tapered sequence u = sqrt(W) (y - m) with demean=False: it divides by n
    instead of sum W, which scales every gamma alike and leaves phi unchanged."""
    W = w_learned * lam**ages
    m = np.sum(W * y_learned) / np.sum(W)
    u = np.sqrt(W) * (y_learned - m)
    a = np.sqrt(W)
    S = np.array([np.sum(u[k:] * u[: len(u) - k]) for k in range(p + 1)])
    P = np.array([np.sum(a[k:] * a[: len(a) - k]) for k in range(p + 1)])
    gamma = S / (P[0] if method == "mle" else P)
    sigma2, phi, *_ = levinson_durbin(gamma, nlags=p, isacov=True)
    phi_yw = None
    if method == "mle":
        phi_yw, _ = yule_walker(u, order=p, method="mle", demean=False, result_object=False)
    return phi, phi_yw


def gap(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = float(np.max(np.abs(a - b)))
    return ab, ab / float(np.max(np.abs(b)))


def run_ls(model: OnlineARLS, y, w) -> np.ndarray:
    return np.array([model.update(y[i], w[i]) for i in range(len(y))])


# --------------------------------------------------------------------------- part 1


def section_no_decay() -> None:
    print("\n## No decay: against AutoReg (OLS) and yule_walker\n")
    print(
        "| p | N | AutoReg abs | rel | floor: AutoReg vs lstsq rel | yule_walker mle abs | rel | adjusted abs | rel | sigma mle rel | max cond(X) | max reciprocal root |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in ORDERS:
        for n in LENGTHS:
            worst = np.zeros(10)
            for seed in SEEDS:
                y, phi = ar_series(p, n, seed)
                X, t = lag_design(y, p)
                floor = gap(
                    np.linalg.lstsq(X, t, rcond=None)[0], AutoReg(y, lags=p, trend="c").fit().params
                )[1]
                ls = OnlineARLS(p, solve_every_row=False)
                yw = OnlineARYW(p)
                for i in range(n):
                    ls.update(y[i], 1.0)
                    yw.update(y[i], 1.0)
                ar = AutoReg(y, lags=p, trend="c").fit().params
                g_ls = gap(ls.solve(), ar)
                phi_m, s2_m = yw.solve("mle")
                rho_m, sig_m = yule_walker(
                    y, order=p, method="mle", demean=True, result_object=False
                )
                phi_a, _ = yw.solve("adjusted")
                rho_a, _ = yule_walker(
                    y, order=p, method="adjusted", demean=True, result_object=False
                )
                g_m, g_a = gap(phi_m, rho_m), gap(phi_a, rho_a)
                g_s = gap([np.sqrt(s2_m)], [sig_m])[1]
                root = np.max(np.abs(np.roots(np.r_[1.0, -phi])))
                worst = np.maximum(worst, [*g_ls, floor, *g_m, *g_a, g_s, np.linalg.cond(X), root])
            print(
                f"| {p} | {n} | "
                + " | ".join(f"{v:.1e}" for v in worst[:8])
                + f" | {worst[8]:.0f} | {worst[9]:.2f} |"
            )


def section_no_shift() -> None:
    print("\n### The same, with the sums kept about zero instead of the first learned value\n")
    print("| p | N | AutoReg rel | yule_walker mle rel |")
    print("|---|---|---|---|")
    for p in ORDERS:
        for n in LENGTHS:
            worst = np.zeros(2)
            for seed in SEEDS:
                y, _ = ar_series(p, n, seed)
                ls, yw = (
                    OnlineARLS(p, solve_every_row=False, shift=False),
                    OnlineARYW(p, shift=False),
                )
                for i in range(n):
                    ls.update(y[i], 1.0)
                    yw.update(y[i], 1.0)
                ar = AutoReg(y, lags=p, trend="c").fit().params
                rho, _ = yule_walker(y, order=p, method="mle", demean=True, result_object=False)
                worst = np.maximum(worst, [gap(ls.solve(), ar)[1], gap(yw.solve("mle")[0], rho)[1]])
            print(f"| {p} | {n} | {worst[0]:.1e} | {worst[1]:.1e} |")


def section_prefix() -> None:
    """Hard rule 2: pred_t equals AutoReg fitted on rows < t, then applied at t."""
    p, n = 5, 400
    y, _ = ar_series(p, n, 7)
    ls = OnlineARLS(p)
    preds = run_ls(ls, y, np.ones(n))
    worst = 0.0
    for t in (2 * p + 2, 100, 399):
        beta = AutoReg(y[:t], lags=p, trend="c").fit().params
        oracle = beta[0] + beta[1:] @ y[t - 1 : t - p - 1 : -1]
        worst = max(worst, abs(preds[t] - oracle) / abs(oracle))
    print(
        f"\nOut of sample: pred_t against AutoReg on rows < t (p = {p}, t = 12, 100, 399): max rel gap {worst:.1e}"
    )


def section_decay() -> None:
    print("\n## Decay: against weighted least squares written longhand, and WLS\n")
    print(
        "| lam | p | N | LS vs lstsq abs | rel | LS vs WLS rel | YW mle vs levinson_durbin abs | rel | YW mle vs yule_walker(tapered) rel | YW adjusted rel |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for lam in (0.999, 0.99):
        for p in ORDERS:
            for n in LENGTHS:
                worst = np.zeros(7)
                for seed in SEEDS:
                    y, _ = ar_series(p, n, seed)
                    w = np.random.default_rng(100 + seed).uniform(0.5, 1.5, n)
                    ls, yw = OnlineARLS(p, lam, solve_every_row=False), OnlineARYW(p, lam)
                    for i in range(n):
                        ls.update(y[i], w[i])
                        yw.update(y[i], w[i])
                    ages = (n - 1 - np.arange(n)).astype(float)
                    b_l, b_sm = wls_longhand(y, ages, w, p, lam)
                    g = gap(ls.solve(), b_l)
                    g_sm = gap(ls.solve(), b_sm)[1]
                    phi_m, _ = yw.solve("mle")
                    phi_a, _ = yw.solve("adjusted")
                    o_m, o_yw = yw_longhand(y, ages, w, p, lam, "mle")
                    o_a, _ = yw_longhand(y, ages, w, p, lam, "adjusted")
                    worst = np.maximum(
                        worst, [*g, g_sm, *gap(phi_m, o_m), gap(phi_m, o_yw)[1], gap(phi_a, o_a)[1]]
                    )
                print(f"| {lam} | {p} | {n} | " + " | ".join(f"{v:.1e}" for v in worst) + " |")


def section_zero_weight() -> None:
    print("\n## Zero-weight rows and gaps\n")
    p, n = 5, 2000
    y, _ = ar_series(p, n, 11)
    rng = np.random.default_rng(12)
    # interleave: zero-weight rows (a value, weight 0) and gaps (NaN, weight 0),
    # the first row of the stream among them
    extra = rng.random(n) < 0.15
    kinds = rng.random(n) < 0.5
    ys, ws, clock_learned = [np.nan], [0.0], []  # first row: a gap
    ys.append(123.0)
    ws.append(0.0)  # second row: zero weight with a wild value
    for i in range(n):
        if extra[i]:
            ys.append(np.nan if kinds[i] else 1e6)
            ws.append(0.0)
        clock_learned.append(len(ys))
        ys.append(y[i])
        ws.append(1.0)
    ys_a, ws_a = np.array(ys), np.array(ws)
    clock_learned = np.array(clock_learned)
    T = len(ys_a)
    print(f"stream: {T} rows, {T - n} of weight 0 (the first two among them, half of them NaN)\n")
    # lam = 1: bit-identical to the stream with those rows dropped
    for cls in (OnlineARLS, OnlineARYW):
        a, b = cls(p), cls(p)
        for i in range(T):
            a.update(ys_a[i], ws_a[i])
        for i in range(n):
            b.update(y[i], 1.0)
        ra = a.solve() if cls is OnlineARLS else a.solve("mle")[0]
        rb = b.solve() if cls is OnlineARLS else b.solve("mle")[0]
        same = np.array_equal(ra, rb) and (cls is OnlineARYW or np.array_equal(a.G, b.G))
        print(f"- {cls.__name__}, lam = 1: equal to the bit to the stream without them: {same}")
    # AutoReg(missing='drop') on the NaN-for-weight-0 series
    y_drop = np.where(ws_a > 0, ys_a, np.nan)
    ar = AutoReg(y_drop, lags=p, trend="c", missing="drop").fit().params
    a = OnlineARLS(p)
    for i in range(T):
        a.update(ys_a[i], ws_a[i])
    print(f"- OnlineARLS vs AutoReg(missing='drop'): rel gap {gap(a.solve(), ar)[1]:.1e}")
    # decay: the clock advances on every row, so ages count the weight-0 rows too
    for lam in (0.999, 0.99):
        ls, yw = OnlineARLS(p, lam), OnlineARYW(p, lam)
        for i in range(T):
            ls.update(ys_a[i], ws_a[i])
            yw.update(ys_a[i], ws_a[i])
        ages = (T - 1 - clock_learned).astype(float)
        b_l, _ = wls_longhand(y, ages, np.ones(n), p, lam)
        o_m, o_yw = yw_longhand(y, ages, np.ones(n), p, lam, "mle")
        print(
            f"- lam = {lam}: LS vs longhand WLS (ages on the clock) rel {gap(ls.solve(), b_l)[1]:.1e}; "
            f"YW rel {gap(yw.solve('mle')[0], o_m)[1]:.1e} (vs yule_walker tapered "
            f"{gap(yw.solve('mle')[0], o_yw)[1]:.1e}); any NaN in state: "
            f"{bool(np.isnan(ls.G).any() or np.isnan(yw.sums).any())}"
        )
    # a stream of only weight-0 rows: no NaN, no fit
    ls, yw = OnlineARLS(p, 0.99), OnlineARYW(p, 0.99)
    for v in (np.nan, 1.0, 2.0):
        ls.update(v, 0.0)
        yw.update(v, 0.0)
    print(
        f"- only weight-0 rows: LS beta all NaN (no fit) {bool(np.isnan(ls.solve()).all())}, "
        f"G finite {bool(np.isfinite(ls.G).all())}; YW gamma NaN (no fit) {bool(np.isnan(yw.gamma()).all())}"
    )


def section_cost() -> None:
    print("\n## Per-row cost (numpy, this machine) and state bytes\n")
    print(
        "| p | LS update+solve us/row | LS update only | YW update us/row | YW update+Levinson | LS state B | YW state B |"
    )
    print("|---|---|---|---|---|---|---|")
    n = 4000
    for p in ORDERS:
        y, _ = ar_series(p, n, 0)
        out = []
        for cls, kw in ((OnlineARLS, {}), (OnlineARLS, {"solve_every_row": False})):
            m = cls(p, 0.999, **kw)
            t0 = time.perf_counter()
            for i in range(n):
                m.update(y[i], 1.0)
            out.append((time.perf_counter() - t0) / n * 1e6)
        ls_bytes = m.nbytes
        m = OnlineARYW(p, 0.999)
        t0 = time.perf_counter()
        for i in range(n):
            m.update(y[i], 1.0)
        out.append((time.perf_counter() - t0) / n * 1e6)
        m = OnlineARYW(p, 0.999)
        t0 = time.perf_counter()
        for i in range(n):
            m.update(y[i], 1.0)
            m.solve()
        out.append((time.perf_counter() - t0) / n * 1e6)
        print(f"| {p} | " + " | ".join(f"{v:.1f}" for v in out) + f" | {ls_bytes} | {m.nbytes} |")


def part_ar() -> None:
    t0 = time.perf_counter()
    section_no_decay()
    section_no_shift()
    section_prefix()
    section_decay()
    section_zero_weight()
    section_cost()
    print(f"\n(part 1 ran in {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["ar"]
    if "ar" in parts:
        part_ar()
