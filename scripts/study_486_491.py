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

Part 2 (task 491): streaming spectral estimates.

* ``StreamingWelch``: a ring of the last ``nperseg`` rows; every ``hop = nperseg -
  noverlap`` rows the ring's segment periodogram is folded into the Welch mean and
  into an exponentially weighted periodogram.
* ``StreamingLombScargle``: the last ``W`` rows of an uneven clock and six running
  sums per frequency (three shared), added on entry, subtracted on leaving, rebuilt
  from the ring every ``W`` rows.

Oracles: ``scipy.signal`` ``welch``, ``periodogram``, ``spectrogram`` (per-segment,
for the weighted mean) and ``lombscargle``; astropy ``LombScargle`` for the
normalisation.

Every series is generated here from a seed; nothing is read from or written to
disk. Run: ``.venv/bin/python scripts/study_486_491.py [ar] [spectral]``. Part 3
(order selection) adds its section to this file.
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


# =========================================================================== part 2
# Task 491: streaming spectral estimates against scipy.signal.

FS = 10.0
SPEC_SEGMENTS = (256, 1024, 4096)
SPEC_N = 32768


def two_sines(n: int, seed: int, fs: float = FS) -> np.ndarray:
    """Stationary: 3 + sin(2 pi 0.73 t) + 0.5 sin(2 pi 2.1 t + 1) + N(0, 1), t = i / fs."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    return (
        3.0
        + np.sin(2 * np.pi * 0.73 * t)
        + 0.5 * np.sin(2 * np.pi * 2.1 * t + 1.0)
        + rng.standard_normal(n)
    )


def taper(name: str, n: int) -> np.ndarray:
    """The periodic window (``get_window(name, n)``, fftbins=True). The window is
    configuration, not state; scipy's values are used so the comparison isolates the
    streaming. Written out, hann is ``0.5 - 0.5 cos(2 pi k / n)``, which differs from
    scipy's (a cosine of ``linspace(-pi, pi)``) by 1 ulp; ``section_welch`` reports it."""
    return scipy.signal.get_window(name, n)


def one_sided(p: np.ndarray, n: int) -> np.ndarray:
    """Double every bin but DC and (n even) Nyquist: scipy's one-sided density."""
    p = p.copy()
    if n % 2 == 0:
        p[1:-1] *= 2.0
    else:
        p[1:] *= 2.0
    return p


def segment_periodogram(
    x: np.ndarray, w: np.ndarray, win: np.ndarray, fs: float
) -> np.ndarray | None:
    """One segment's one-sided density. ``w`` are data weights: a sample's amplitude is
    ``win * sqrt(w)``; the constant detrend removes the w-weighted mean; the scale is
    ``1 / (fs * sum (win^2 w))``. At w = 1 this is scipy's periodogram(detrend='constant').
    A weight-0 sample's value is never multiplied (it may be NaN). None if sum w = 0."""
    live = w > 0.0
    sw = float(np.sum(w[live]))
    if not sw > 0.0:
        return None
    xx = np.where(live, x, 0.0)
    m = float(np.sum(w[live] * xx[live])) / sw
    a = win * np.sqrt(np.where(live, w, 0.0))
    spec = np.abs(np.fft.rfft(a * (xx - m))) ** 2 / (fs * float(np.sum(a * a)))
    return one_sided(spec, len(x))


class StreamingWelch:
    """Welch over a ring of the last ``nperseg`` rows, hop = nperseg - noverlap.

    Every row writes its value and weight into the ring (a zero-weight row holds its
    slot on the clock and contributes nothing). Once the ring is full, every ``hop``
    rows the ring's segment periodogram is folded into two accumulators:
    the mean (``P_sum / M``, scipy's welch) and the exponentially weighted one
    (``P_ew = beta P_ew + P_seg``, ``Z = beta Z + 1``, read ``P_ew / Z``)."""

    def __init__(self, nperseg: int, noverlap: int, window: str, fs: float = FS, beta: float = 0.9):
        self.n, self.hop, self.fs, self.beta = nperseg, nperseg - noverlap, fs, beta
        self.win = taper(window, nperseg)  # configuration, recomputable; not state
        self.x = np.zeros(nperseg)
        self.w = np.zeros(nperseg)
        self.pos = 0  # next slot
        self.rows = 0  # rows seen (the clock)
        k = nperseg // 2 + 1
        self.p_sum = np.zeros(k)
        self.m = 0
        self.p_ew = np.zeros(k)
        self.z = 0.0
        self.last: np.ndarray | None = None  # the latest segment's periodogram (for the check only)

    def update(self, x: float, w: float = 1.0) -> None:
        self.x[self.pos] = x
        self.w[self.pos] = w
        self.pos = (self.pos + 1) % self.n
        self.rows += 1
        if self.rows >= self.n and (self.rows - self.n) % self.hop == 0:
            xs = np.concatenate((self.x[self.pos :], self.x[: self.pos]))
            ws = np.concatenate((self.w[self.pos :], self.w[: self.pos]))
            p = segment_periodogram(xs, ws, self.win, self.fs)
            self.p_ew *= self.beta
            self.z *= self.beta
            self.last = p
            if p is not None:
                self.p_sum += p
                self.m += 1
                self.p_ew += p
                self.z += 1.0

    def welch(self) -> np.ndarray:
        return self.p_sum / self.m if self.m else np.full_like(self.p_sum, np.nan)

    def ew(self) -> np.ndarray:
        return self.p_ew / self.z if self.z > 0 else np.full_like(self.p_ew, np.nan)

    @property
    def nbytes(self) -> int:
        # ring values + weights, mean sum, EW sum, M, Z, pos, rows
        return self.x.nbytes + self.w.nbytes + self.p_sum.nbytes + self.p_ew.nbytes + 4 * 8


class StreamingLombScargle:
    """Lomb-Scargle over the last ``W`` rows of an uneven clock, at fixed angular
    frequencies ``omega``. Per frequency six sums (weights normalised at read):
    ``Sc = sum w c, Ss = sum w s, Scc = sum w c^2, Scs = sum w c s, Syc = sum w y c,
    Sys = sum w y s`` with ``c, s = cos, sin(omega (t - o))``; three shared:
    ``Sw, Sy = sum w y, Syy = sum w y^2``; and the count of weighted rows. A row adds
    its terms; the row leaving the window subtracts its terms (its trig recomputed
    from the ring). Every ``W`` rows the sums are recomputed from the ring, the
    origin ``o`` rebased to the oldest time, which bounds both the subtraction's
    drift and the size of the phase argument."""

    def __init__(self, omega: np.ndarray, window: int, recompute: bool = True):
        self.om = np.asarray(omega, float)
        self.W, self.recompute = window, recompute
        self.t = np.zeros(window)
        self.y = np.zeros(window)
        self.w = np.zeros(window)
        self.pos = self.filled = self.rows = 0
        self.o = 0.0
        self.origin_set = False
        f = len(self.om)
        self.s = np.zeros((6, f))
        self.g = np.zeros(3)
        self.cnt = 0

    def _terms(self, t: float, y: float, w: float) -> np.ndarray:
        arg = self.om * (t - self.o)
        c, s = np.cos(arg), np.sin(arg)
        return w * np.array((c, s, c * c, c * s, y * c, y * s))

    def update(self, t: float, y: float, w: float = 1.0) -> None:
        if not self.origin_set:
            self.o, self.origin_set = t, True
        if self.filled == self.W:  # the leaving row
            wo = self.w[self.pos]
            if wo > 0.0:
                to, yo = self.t[self.pos], self.y[self.pos]
                self.s -= self._terms(to, yo, wo)
                self.g -= (wo, wo * yo, wo * yo * yo)
                self.cnt -= 1
                if self.cnt == 0:  # an empty window drops the subtraction's residue
                    self.s[:] = 0.0
                    self.g[:] = 0.0
        else:
            self.filled += 1
        self.t[self.pos], self.y[self.pos], self.w[self.pos] = t, y, w
        self.pos = (self.pos + 1) % self.W
        self.rows += 1
        if w > 0.0:  # a zero-weight row holds a slot of the window and adds nothing
            self.s += self._terms(t, y, w)
            self.g += (w, w * y, w * y * y)
            self.cnt += 1
        if self.recompute and self.rows % self.W == 0:
            self._rebuild()

    def _rebuild(self) -> None:
        idx = (np.arange(self.filled) + (self.pos - self.filled)) % self.W
        t, y, w = self.t[idx], self.y[idx], self.w[idx]
        live = w > 0.0
        t, y, w = t[live], y[live], w[live]
        self.o = float(t[0]) if len(t) else self.o
        arg = np.outer(t - self.o, self.om)
        c, s = np.cos(arg), np.sin(arg)
        wc = w[:, None]
        self.s = np.array(
            [
                (wc * c).sum(0),
                (wc * s).sum(0),
                (wc * c * c).sum(0),
                (wc * c * s).sum(0),
                (wc * y[:, None] * c).sum(0),
                (wc * y[:, None] * s).sum(0),
            ]
        )
        self.g = np.array([w.sum(), (w * y).sum(), (w * y * y).sum()])
        self.cnt = int(live.sum())

    def power(self, normalize: str = "power", floating_mean: bool = False) -> np.ndarray:
        """scipy's lombscargle on the window's weighted rows: the weighted projection of y
        on (cos, sin) -- on (1, cos, sin) with ``floating_mean`` -- which is what the
        phase offset tau diagonalises. 'power': (N / 2) v' M^-1 v; 'normalize':
        v' M^-1 v / YY, with the weights normalised to sum 1."""
        sw = self.g[0]
        if self.cnt == 0 or not sw > 0.0:  # no weighted row in the window: no fit
            return np.full(len(self.om), np.nan)
        C, S, CC, CS, YC, YS = self.s / sw
        Y, YY = self.g[1] / sw, self.g[2] / sw
        SS = 1.0 - CC
        if floating_mean:
            CC, SS, CS = CC - C * C, SS - S * S, CS - C * S
            YC, YS, YY = YC - Y * C, YS - Y * S, YY - Y * Y
        det = CC * SS - CS * CS
        # A column that vanishes on the rows (the sine at the Nyquist of an even clock)
        # leaves a one-column fit; scipy reaches the same by clamping CC, SS at epsneg.
        tol = 1e-10 * (CC + SS)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = (SS * YC * YC - 2.0 * CS * YC * YS + CC * YS * YS) / det
            q = np.where(tol >= SS, YC * YC / CC, np.where(tol >= CC, YS * YS / SS, q))
        if normalize == "power":
            return 0.5 * self.cnt * q
        return q / YY

    @property
    def nbytes(self) -> int:
        # ring t, y, w; six sums per frequency; three shared sums; pos, filled, rows, count, origin
        return self.t.nbytes + self.y.nbytes + self.w.nbytes + self.s.nbytes + self.g.nbytes + 5 * 8


def spec_gap(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = float(np.max(np.abs(a - b)))
    return ab, ab / float(np.max(np.abs(b)))


def section_welch() -> None:
    print("\n## Streaming Welch and the segment periodogram vs scipy.signal\n")
    print(
        f"Signal: two sines + N(0,1) + 3, fs = {FS}, N = {SPEC_N}, seed 0; reads at 12 checkpoints (scipy welch on x[:t]).\n"
    )
    k = np.arange(4096)
    print(
        f"(hann written out vs scipy's get_window at n = 4096: max gap {np.max(np.abs(0.5 - 0.5 * np.cos(2 * np.pi * k / 4096) - taper('hann', 4096))):.1e}.)\n"
    )
    print(
        "| nperseg | overlap | window | segments | welch abs | welch rel | periodogram abs | periodogram rel | EW (beta=0.9) vs spectrogram abs | rel |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    x = two_sines(SPEC_N, 0)
    worst = {}
    for nps in SPEC_SEGMENTS:
        for ov in (0, nps // 2):
            for wname in ("hann", "boxcar"):
                m = StreamingWelch(nps, ov, wname)
                checks = set(np.linspace(nps, SPEC_N, 12).astype(int)) | {SPEC_N}
                g = np.zeros(6)
                for i in range(SPEC_N):
                    m.update(x[i])
                    t = i + 1
                    if t in checks:
                        _, pw = scipy.signal.welch(
                            x[:t],
                            FS,
                            window=wname,
                            nperseg=nps,
                            noverlap=ov,
                            detrend="constant",
                            scaling="density",
                        )
                        a, r = spec_gap(m.welch(), pw)
                        g[0], g[1] = max(g[0], a), max(g[1], r)
                        # the latest segment is x[s:s+nps], s the last segment start <= t - nps
                        s0 = (t - nps) // m.hop * m.hop
                        _, pp = scipy.signal.periodogram(
                            x[s0 : s0 + nps],
                            FS,
                            window=wname,
                            detrend="constant",
                            scaling="density",
                        )
                        a, r = spec_gap(m.last, pp)
                        g[2], g[3] = max(g[2], a), max(g[3], r)
                        _, _, sx = scipy.signal.spectrogram(
                            x[:t],
                            FS,
                            window=wname,
                            nperseg=nps,
                            noverlap=ov,
                            detrend="constant",
                            scaling="density",
                            mode="psd",
                        )
                        k = sx.shape[1]
                        wt = m.beta ** np.arange(k - 1, -1, -1)
                        a, r = spec_gap(m.ew(), sx @ wt / wt.sum())
                        g[4], g[5] = max(g[4], a), max(g[5], r)
                worst[(nps, ov, wname)] = g
                print(
                    f"| {nps} | {ov} | {wname} | {m.m} | "
                    + " | ".join(f"{v:.1e}" for v in g)
                    + " |"
                )


def section_welch_zero_weight() -> None:
    print("\n## Welch with zero-weight rows (10% weight 0, half NaN; others U(0.5, 1.5))\n")
    n, nps = 8192, 256
    x = two_sines(n, 1)
    rng = np.random.default_rng(7)
    w = rng.uniform(0.5, 1.5, n)
    z = rng.choice(n, n // 10, replace=False)
    w[z] = 0.0
    x = x.copy()
    x[z[: len(z) // 2]] = np.nan
    w[:3] = 0.0  # the stream opens on zero-weight rows
    x[0] = np.nan
    # one whole segment of zero weight
    w[2048 : 2048 + nps] = 0.0
    print(
        "| overlap | window | welch rel vs scipy periodogram per segment (window = win*sqrt(w), detrend=False on x - m_w) | EW rel | NaN in state | segments kept / seen |"
    )
    print("|---|---|---|---|---|---|")
    for ov in (0, nps // 2):
        for wname in ("hann", "boxcar"):
            m = StreamingWelch(nps, ov, wname)
            for i in range(n):
                m.update(x[i], w[i])
            hop = nps - ov
            segs, wts = [], []
            starts = range(0, n - nps + 1, hop)
            for j, s0 in enumerate(starts):
                xs, ws = x[s0 : s0 + nps], w[s0 : s0 + nps]
                live = ws > 0
                if not live.any():
                    continue
                xx = np.where(live, xs, 0.0)
                mw = np.sum(ws[live] * xx[live]) / np.sum(ws[live])
                _, pp = scipy.signal.periodogram(
                    xx - mw,
                    FS,
                    window=taper(wname, nps) * np.sqrt(ws),
                    detrend=False,
                    scaling="density",
                )
                segs.append(pp)
                wts.append(m.beta ** (len(starts) - 1 - j))
            segs = np.array(segs)
            _, rw = spec_gap(m.welch(), segs.mean(0))
            wts = np.array(wts)
            _, re = spec_gap(m.ew(), wts @ segs / wts.sum())
            nan = any(np.isnan(a).any() for a in (m.p_sum, m.p_ew, np.array([m.z])))
            print(f"| {ov} | {wname} | {rw:.1e} | {re:.1e} | {nan} | {m.m} / {len(starts)} |")
    # w = 1 everywhere with weights passed: identical to the unweighted path, to the bit
    m1, m2 = StreamingWelch(nps, nps // 2, "hann"), StreamingWelch(nps, nps // 2, "hann")
    xa = two_sines(4096, 2)
    for v in xa:
        m1.update(v)
        m2.update(v, 1.0)
    print(
        f"\nUnit weights passed explicitly equal the default to the bit: {np.array_equal(m1.welch(), m2.welch())}."
    )


def ls_oracle(t, y, w, om, normalize, floating_mean):
    live = w > 0
    return scipy.signal.lombscargle(
        t[live], y[live], om, normalize=normalize, weights=w[live], floating_mean=floating_mean
    )


def section_lomb_scargle() -> None:
    from astropy.timeseries import LombScargle

    print("\n## Lomb-Scargle normalisation: scipy vs astropy (slow method), 2000 jittered rows\n")
    rng = np.random.default_rng(3)
    n, dt = 2000, 1.0 / FS
    t = (np.arange(n) + rng.uniform(-0.3, 0.3, n)) * dt
    y = (
        3.0
        + np.sin(2 * np.pi * 0.73 * t)
        + 0.5 * np.sin(2 * np.pi * 2.1 * t + 1.0)
        + rng.standard_normal(n)
    )
    f = np.linspace(0.05, FS / 2, 64)
    om = 2 * np.pi * f
    print("| scipy lombscargle | astropy LombScargle | rel gap |")
    print("|---|---|---|")
    for fm in (False, True):
        for sn, an in (("power", "psd"), ("normalize", "standard")):
            ps = scipy.signal.lombscargle(t, y, om, normalize=sn, floating_mean=fm)
            pa = LombScargle(t, y, fit_mean=fm, center_data=fm, normalization=an).power(
                f, method="slow"
            )
            print(
                f"| normalize='{sn}', floating_mean={fm} | normalization='{an}', fit_mean={fm}, center_data={fm} | {spec_gap(ps, pa)[1]:.1e} |"
            )
    # astropy's default center_data=True with fit_mean=False vs scipy on y - mean
    ps = scipy.signal.lombscargle(t, y - y.mean(), om, normalize="normalize")
    pa = LombScargle(t, y, fit_mean=False, center_data=True).power(f, method="slow")
    print(
        f"| normalize='normalize' on y - mean(y) | astropy defaults but fit_mean=False | {spec_gap(ps, pa)[1]:.1e} |"
    )
    ps = scipy.signal.lombscargle(t, y, om, normalize="normalize", floating_mean=True)
    pa = LombScargle(t, y).power(f, method="slow")
    print(
        f"| normalize='normalize', floating_mean=True | astropy defaults (standard, fit_mean, center_data) | {spec_gap(ps, pa)[1]:.1e} |"
    )

    print(
        "\n## Streaming Lomb-Scargle over the last W rows vs scipy.signal.lombscargle on the window's rows\n"
    )
    print(
        "Clock t_i = (i + u_i) / fs, u_i ~ U(-0.3, 0.3) (30% jitter), 64 frequencies 0.05..5 Hz; N = 4 W + 37 rows; 9 reads."
    )
    print(
        "Gap: worst over the reads and the four (normalize, floating_mean) forms. 'raw t' gives scipy the times as they"
    )
    print("came; 'rebased' gives it t - t_oldest, the stream's own origin.\n")
    print(
        "Columns 4 to 7 leave out the last frequency, the Nyquist (5 Hz), which has its own two columns.\n"
    )
    print(
        "| W | jitter | recompute every W | rel vs scipy (raw t) | abs | rel vs scipy (rebased) | no-recompute rel (raw t) | Nyquist rel vs scipy | Nyquist rel vs written out |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for W in SPEC_SEGMENTS:
        for jit in (0.3, 0.0):
            n = 4 * W + 37
            rng = np.random.default_rng(11)
            t = (np.arange(n) + rng.uniform(-jit, jit, n)) * dt
            y = (
                3.0
                + np.sin(2 * np.pi * 0.73 * t)
                + 0.5 * np.sin(2 * np.pi * 2.1 * t + 1.0)
                + rng.standard_normal(n)
            )
            w = np.ones(n)
            reads = set(np.linspace(W // 2, n, 9).astype(int))
            res = []
            for rec in (True, False):
                m = StreamingLombScargle(om, W, recompute=rec)
                g = np.zeros(5)
                for i in range(n):
                    m.update(t[i], y[i], w[i])
                    if i + 1 in reads:
                        lo = max(0, i + 1 - W)
                        tw, yw, ww = t[lo : i + 1], y[lo : i + 1], w[lo : i + 1]
                        for sn in ("power", "normalize"):
                            for fm in (False, True):
                                ours = m.power(sn, fm)
                                ref = ls_oracle(tw, yw, ww, om, sn, fm)
                                scale = np.max(np.abs(ref))
                                d = np.abs(ours - ref)
                                g[0], g[1] = (
                                    max(g[0], d[:-1].max() / scale),
                                    max(g[1], d[:-1].max()),
                                )
                                ref2 = ls_oracle(tw - tw[0], yw, ww, om, sn, fm)
                                g[2] = max(
                                    g[2], np.abs(ours - ref2)[:-1].max() / np.max(np.abs(ref2))
                                )
                                g[3] = max(g[3], d[-1] / scale)
                                if (
                                    jit == 0.0
                                ):  # the Nyquist written out: the cosine column alone, (-1)^i
                                    c, yy, sw = (
                                        (-1.0) ** np.arange(lo, i + 1),
                                        yw.copy(),
                                        ww / ww.sum(),
                                    )
                                    if fm:
                                        c, yy = c - sw @ c, yy - sw @ yy
                                    q = (sw @ (yy * c)) ** 2 / (sw @ (c * c))
                                    ex = (
                                        0.5 * len(yw) * q if sn == "power" else q / (sw @ (yy * yy))
                                    )
                                    g[4] = max(g[4], abs(ours[-1] - ex) / scale)
                res.append(g)
            nyq_ex = f"{res[0][4]:.1e}" if jit == 0.0 else "n/a"
            print(
                f"| {W} | {jit} | yes | {res[0][0]:.1e} | {res[0][1]:.1e} | {res[0][2]:.1e} | {res[1][0]:.1e} | {res[0][3]:.1e} | {nyq_ex} |"
            )

    print("\n### Zero-weight rows and a long stream (W = 256)\n")
    W, n = 256, 20000
    rng = np.random.default_rng(5)
    t = (np.arange(n) + rng.uniform(-0.3, 0.3, n)) * dt
    y = 3.0 + np.sin(2 * np.pi * 0.73 * t) + rng.standard_normal(n)
    w = rng.uniform(0.5, 1.5, n)
    z = rng.choice(n, n // 10, replace=False)
    w[z] = 0.0
    y[z[::2]] = np.nan
    w[:2], y[0] = 0.0, np.nan
    w[5000 : 5000 + W] = 0.0  # one whole window of weight 0
    for rec in (True, False):
        m = StreamingLombScargle(om, W, recompute=rec)
        worst, nan_state, empty_nan = 0.0, False, None
        for i in range(n):
            m.update(t[i], y[i], w[i])
            nan_state |= bool(np.isnan(m.s).any() or np.isnan(m.g).any())
            if i + 1 == 5000 + W:
                empty_nan = bool(np.isnan(m.power()).all())
            if (i + 1) % 997 == 0 and i + 1 != 5000 + W:
                lo = max(0, i + 1 - W)
                if (w[lo : i + 1] > 0).any():
                    for sn in ("power", "normalize"):
                        for fm in (False, True):
                            worst = max(
                                worst,
                                spec_gap(
                                    m.power(sn, fm),
                                    ls_oracle(
                                        t[lo : i + 1], y[lo : i + 1], w[lo : i + 1], om, sn, fm
                                    ),
                                )[1],
                            )
        print(
            f"- recompute={rec}: {n} rows, 10% weight 0 (half NaN), weights U(0.5, 1.5): worst rel vs scipy (weights=w on the w > 0 rows) {worst:.1e}; NaN in the sums: {nan_state}; a window of only weight-0 rows reads NaN (no fit): {empty_nan}."
        )


def section_spectral_cost() -> None:
    print("\n## Cost (numpy, this machine) and state bytes\n")
    print(
        "| nperseg / W | overlap | Welch state B | Welch us/row (amortised, hop rows per FFT) | one segment (FFT) us | Welch read us | LS state B (F=64) | LS us/row | LS read us | LS rebuild us (every W rows) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    om = 2 * np.pi * np.linspace(0.05, FS / 2, 64)
    for nps in SPEC_SEGMENTS:
        n = max(8 * nps, 16384)
        x = two_sines(n, 0)
        for ov in (0, nps // 2):
            m = StreamingWelch(nps, ov, "hann")
            t0 = time.perf_counter()
            for i in range(n):
                m.update(x[i])
            per_row = (time.perf_counter() - t0) / n * 1e6
            ws = np.ones(nps)
            t0 = time.perf_counter()
            for _ in range(200):
                segment_periodogram(x[:nps], ws, m.win, FS)
            seg = (time.perf_counter() - t0) / 200 * 1e6
            t0 = time.perf_counter()
            for _ in range(2000):
                m.welch()
            rd = (time.perf_counter() - t0) / 2000 * 1e6
            if ov == 0:
                ls = StreamingLombScargle(om, nps)
                tt = np.arange(n) / FS
                t0 = time.perf_counter()
                for i in range(n):
                    ls.update(tt[i], x[i])
                ls_row = (time.perf_counter() - t0) / n * 1e6
                t0 = time.perf_counter()
                for _ in range(2000):
                    ls.power("power", True)
                ls_rd = (time.perf_counter() - t0) / 2000 * 1e6
                t0 = time.perf_counter()
                for _ in range(20):
                    ls._rebuild()
                ls_rb = (time.perf_counter() - t0) / 20 * 1e6
                lsc = f"{ls.nbytes} | {ls_row:.1f} | {ls_rd:.1f} | {ls_rb:.0f}"
            else:
                lsc = " | | | "
            print(f"| {nps} | {ov} | {m.nbytes} | {per_row:.2f} | {seg:.1f} | {rd:.2f} | {lsc} |")


def part_spectral() -> None:
    t0 = time.perf_counter()
    section_welch()
    section_welch_zero_weight()
    section_lomb_scargle()
    section_spectral_cost()
    print(f"\n(part 2 ran in {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["ar"]
    if "ar" in parts:
        part_ar()
    if "spectral" in parts:
        part_spectral()
