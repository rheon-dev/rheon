"""An online VAR(p) with Granger-causality tests, numpy only (PLAN task 495, study).

Model
-----
For a ``k``-variate stream ``y_t`` the VAR(p) with a constant is

    y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,    u_t ~ (0, Sigma_u).

Write ``x_t = [y_{t-1}; ...; y_{t-p}]`` (length ``m = k p``, lag 1 first, variables
in order within a lag: the column order of ``statsmodels``' ``VAR`` design after
its constant) and ``B`` for the ``m x k`` slope matrix, so ``y_t = c + B' x_t + u_t``.
The model keeps the last ``p`` observations itself, so a row is just ``y_t``: the
first ``p`` rows of a stream (and the ``p`` rows after any row with a non-finite
value) only fill the lag buffer and are not targets.

Update (mean form, exponentially weighted, one pass)
-----------------------------------------------------
With decay ``lam`` in (0, 1] and row weight ``w >= 0``, for a target row:

    W_t   = lam W_{t-1} + w                       (weight sum)
    V_t   = lam^2 V_{t-1} + w^2                   (sum of squared weights)
    a     = w / W_t                               (0 when W_t = 0: guarded 0/0)
    dx    = x_t - xbar_{t-1},  dy = y_t - ybar_{t-1}
    xbar_t = xbar_{t-1} + a dx,  ybar_t = ybar_{t-1} + a dy
    Sxx_t = lam Sxx_{t-1} + w (1 - a) dx dx'
    Sxy_t = lam Sxy_{t-1} + w (1 - a) dx dy'
    Syy_t = lam Syy_{t-1} + w (1 - a) dy dy'

These are the weighted means and centred co-moments of ``(x, y)``; with
``lam = 1`` and ``w = 1`` they are the plain sample ones, so the fit below is
ordinary least squares with an intercept on the rows seen so far.

Fit (read from the sums, any time)
----------------------------------
    B      = Sxx^{-1} Sxy                         (Cholesky of Sxx)
    c      = ybar - B' xbar
    SSE    = Syy - Sxy' B
    n_eff  = W^2 / V                              (Kish; the row count when w = 1, lam = 1)
    df     = n_eff - (m + 1)
    Sigma_u = SSE n_eff / (W df)                  (statsmodels' ``sigma_u``)
    Sigma_mle = SSE / W                           (statsmodels' ``sigma_u_mle``)

With no decay and unit weights ``W = n_eff = n`` and these are statsmodels'
``SSE / df`` and ``SSE / n``. Under decay ``SSE / W`` is the weighted mean squared
residual, which estimates the noise covariance; ``SSE / df`` would not (it
estimates ``Sigma (V / W)``, half of it at steady state, ``V / W -> 1 / (1 + lam)``).

The inverse of the weighted ``Z'Z`` (``Z = [1, X]``) is block form:
``(Z'Z)^{-1}`` has lag block ``Sxx^{-1}``, cross block ``-Sxx^{-1} xbar`` and
constant entry ``1/W + xbar' Sxx^{-1} xbar``, and

    cov_params = (W / n_eff) (Z'Z)^{-1} kron Sigma_u

the sandwich ``(Z'Ω Z)^{-1} Z'Ω^2 Z (Z'Ω Z)^{-1}`` for homoskedastic noise when the
weights ``Ω`` are independent of the regressors (``Z'Ω^2 Z ~ (V / W) Z'Ω Z``): a
decayed fit has the variance of ``n_eff`` rows, not of ``W``. The factor is 1 with
no decay, so this is statsmodels' ``cov_params`` there; it is ``SSE / df`` times
``(Z'Z)^{-1}`` either way, and so is the Wald statistic below.

Granger causality (Wald, as ``VARResults.test_causality``)
-----------------------------------------------------------
H0: the lags of the variables ``J`` (causing) have zero coefficients in the
equations ``E`` (caused). With ``R`` the lag columns of ``J`` (``q = p |J| |E|``
restrictions), ``G = (Sxx^{-1})_{RR}`` and ``beta = B[R, E]``, the Wald statistic
``(C b)' [C (Z'Z)^{-1} kron Sigma_u C']^{-1} (C b)`` reduces, because the
restricted block of a Kronecker product is the Kronecker product of the blocks, to

    lambda_W = trace( Sigma_u[E, E]^{-1} beta' G^{-1} beta )

which needs only a ``|R| x |R|`` and an ``|E| x |E|`` solve. ``kind="wald"``
refers it to chi2(q); ``kind="f"`` (statsmodels' default) uses ``lambda_W / q``
against F(q, k df), statsmodels' (Lutkepohl's) denominator of ``k`` times the
residual degrees of freedom.

For one caused equation, ``lambda_W = (SSR_r - SSR_u) / (SSR_u / df)`` (the
restricted fit is never run), so ``grangercausalitytests``' single-equation
statistics follow from the same sums: ``ssr_f = lambda_W / q`` against
F(q, df), ``ssr_chi2 = n lambda_W / df`` and ``lr = n log(1 + lambda_W / df)``
against chi2(q).

State size
----------
``p k`` lag buffer, ``m + k`` means, ``m^2 + m k + k^2`` co-moments and three
scalars: ``8 (p k + m + k + m^2 + m k + k^2 + 3)`` bytes with ``m = k p``, bounded
by the model's shape, never by the stream's length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class OnlineVAR:
    """An exponentially weighted VAR(p) with a constant, fed one row at a time."""

    k: int
    p: int
    lam: float = 1.0
    min_weight: float = 0.0
    # State.
    lags: np.ndarray = field(init=False)  # (p, k), row 0 is y_{t-1}
    n_seen: int = field(init=False, default=0)  # finite rows since the last break
    w_sum: float = field(init=False, default=0.0)
    w2_sum: float = field(init=False, default=0.0)
    xbar: np.ndarray = field(init=False)
    ybar: np.ndarray = field(init=False)
    sxx: np.ndarray = field(init=False)
    sxy: np.ndarray = field(init=False)
    syy: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.k < 1 or self.p < 1:
            raise ValueError("k and p must be at least 1")
        if not 0.0 < self.lam <= 1.0:
            raise ValueError("lam must be in (0, 1]")
        m = self.k * self.p
        self.lags = np.full((self.p, self.k), np.nan)
        self.xbar = np.zeros(m)
        self.ybar = np.zeros(self.k)
        self.sxx = np.zeros((m, m))
        self.sxy = np.zeros((m, self.k))
        self.syy = np.zeros((self.k, self.k))

    @property
    def m(self) -> int:
        return self.k * self.p

    def state_bytes(self) -> int:
        """Bytes of f64 state (see the module docstring)."""
        m, k, p = self.m, self.k, self.p
        return 8 * (p * k + m + k + m * m + m * k + k * k + 3)

    # ------------------------------------------------------------------ update
    def _x(self) -> np.ndarray:
        return self.lags.reshape(-1)  # [y_{t-1}, ..., y_{t-p}]

    def step(self, y: np.ndarray, w: float = 1.0) -> tuple[np.ndarray, float]:
        """Predict ``y_t`` from the fit so far, then learn the row.

        Returns ``(pred, weight_sum)``: ``pred`` is computed before this row's
        update (out of sample by construction; NaN while there is no fit) and
        ``weight_sum`` is ``W`` before this row's update and before its decay.
        """
        y = np.asarray(y, dtype=float)
        weight_before = self.w_sum
        x = self._x()
        has_x = self.n_seen >= self.p
        pred = self.predict_x(x) if has_x else np.full(self.k, np.nan)
        if np.all(np.isfinite(y)):
            if has_x:
                self._learn(x, y, float(w))
            self.n_seen += 1
        else:
            self.n_seen = 0  # the next p rows have a missing lag
        self.lags[1:] = self.lags[:-1].copy()
        self.lags[0] = y
        return pred, weight_before

    def _learn(self, x: np.ndarray, y: np.ndarray, w: float) -> None:
        lam = self.lam
        w_new = lam * self.w_sum + w
        self.w2_sum = lam * lam * self.w2_sum + w * w
        if w_new > 0.0:
            a = w / w_new
            dx = x - self.xbar
            dy = y - self.ybar
            self.xbar = self.xbar + a * dx
            self.ybar = self.ybar + a * dy
            g = w * (1.0 - a)
            self.sxx = lam * self.sxx + g * np.outer(dx, dx)
            self.sxy = lam * self.sxy + g * np.outer(dx, dy)
            self.syy = lam * self.syy + g * np.outer(dy, dy)
        else:  # a zero-weight first row: advance the clock, learn nothing
            self.sxx = lam * self.sxx
            self.sxy = lam * self.sxy
            self.syy = lam * self.syy
        self.w_sum = w_new

    # --------------------------------------------------------------------- fit
    def n_eff(self) -> float:
        return self.w_sum * self.w_sum / self.w2_sum if self.w2_sum > 0.0 else 0.0

    def _ready(self, p: int | None = None) -> bool:
        m = self.k * (self.p if p is None else p)
        return self.w_sum > 0.0 and self.w_sum >= self.min_weight and self.n_eff() > m

    def _block(self, p: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lag-order ``p`` sub-model: the leading ``k p`` lag columns of the sums."""
        m = self.k * (self.p if p is None else p)
        return self.sxx[:m, :m], self.sxy[:m], self.xbar[:m]

    def slopes(self, p: int | None = None) -> np.ndarray:
        sxx, sxy, _ = self._block(p)
        chol = np.linalg.cholesky(sxx)
        return np.linalg.solve(chol.T, np.linalg.solve(chol, sxy))

    def params(self, p: int | None = None) -> np.ndarray:
        """``(1 + k p) x k``: constant row, then ``A_1', ..., A_p'`` (statsmodels' ``params``)."""
        b = self.slopes(p)
        _, _, xbar = self._block(p)
        return np.vstack([self.ybar - b.T @ xbar, b])

    def sse(self, p: int | None = None) -> np.ndarray:
        _, sxy, _ = self._block(p)
        return self.syy - sxy.T @ self.slopes(p)

    def df_resid(self, p: int | None = None) -> float:
        return self.n_eff() - (self.k * (self.p if p is None else p) + 1)

    def sigma_u(self, p: int | None = None) -> np.ndarray:
        return self.sse(p) * (self.n_eff() / self.w_sum) / self.df_resid(p)

    def sigma_u_mle(self, p: int | None = None) -> np.ndarray:
        return self.sse(p) / self.w_sum

    def _coef_scale(self) -> float:
        """``W / n_eff``: the factor between ``Sigma_u`` and a decayed fit's variance."""
        return self.w_sum / self.n_eff()

    def zz_inv(self) -> np.ndarray:
        """``(Z'Z)^{-1}`` of the weighted design ``Z = [1, X]``, from the centred sums."""
        sinv = np.linalg.inv(self.sxx)
        sx = sinv @ self.xbar
        m = self.m
        out = np.empty((m + 1, m + 1))
        out[0, 0] = 1.0 / self.w_sum + self.xbar @ sx
        out[0, 1:] = -sx
        out[1:, 0] = -sx
        out[1:, 1:] = sinv
        return out

    def cov_params(self) -> np.ndarray:
        return np.kron(self.zz_inv(), self._coef_scale() * self.sigma_u())

    def predict_x(self, x: np.ndarray) -> np.ndarray:
        if not self._ready():
            return np.full(self.k, np.nan)
        prm = self.params()
        return prm[0] + prm[1:].T @ x

    def info_criteria(self, p: int) -> dict[str, float]:
        """AIC/BIC/HQIC/FPE of the VAR(``p``), ``p <= self.p``, on the same rows.

        The sums of a VAR(pmax) hold every VAR(p <= pmax) as their leading block,
        on the rows ``pmax..T`` (``VAR.select_order``'s common sample).
        """
        n = self.n_eff()
        k = self.k
        free = p * k * k + k
        ld = np.linalg.slogdet(self.sigma_u_mle(p))[1]
        df_model = k * p + 1
        return {
            "aic": ld + 2.0 / n * free,
            "bic": ld + np.log(n) / n * free,
            "hqic": ld + 2.0 * np.log(np.log(n)) / n * free,
            "fpe": ((n + df_model) / self.df_resid(p)) ** k * np.exp(ld),
        }

    # ----------------------------------------------------------------- Granger
    def test_causality(
        self, caused: int | list[int], causing: int | list[int] | None = None, kind: str = "f"
    ) -> dict[str, float]:
        """Granger Wald test from the sums (see the module docstring)."""
        e_idx = [caused] if isinstance(caused, int) else list(caused)
        if causing is None:
            j_idx = [i for i in range(self.k) if i not in e_idx]
        else:
            j_idx = [causing] if isinstance(causing, int) else list(causing)
        r_idx = [lag * self.k + j for lag in range(self.p) for j in j_idx]
        q = len(r_idx) * len(e_idx)
        b = self.slopes()
        beta = b[np.ix_(r_idx, e_idx)]
        g = np.linalg.inv(self.sxx)[np.ix_(r_idx, r_idx)]
        sig = self._coef_scale() * self.sigma_u()[np.ix_(e_idx, e_idx)]
        wald = float(np.trace(np.linalg.solve(sig, beta.T @ np.linalg.solve(g, beta))))
        if kind == "wald":
            return {"statistic": wald, "pvalue": float(stats.chi2.sf(wald, q)), "df": q}
        if kind == "f":
            df2 = self.k * self.df_resid()
            f = wald / q
            return {"statistic": f, "pvalue": float(stats.f.sf(f, q, df2)), "df": (q, df2)}
        # Single-equation forms, as ``statsmodels.tsa.stattools.grangercausalitytests``
        # (one caused equation). There ``wald = (SSR_r - SSR_u) / (SSR_u / df)``, so
        # every one of its statistics is a function of ``wald``, ``df`` and ``n``.
        if len(e_idx) != 1:
            raise ValueError(f"kind {kind!r} tests one caused equation")
        df, n = self.df_resid(), self.n_eff()
        if kind == "ssr_f":
            f = wald / q
            return {"statistic": f, "pvalue": float(stats.f.sf(f, q, df)), "df": (q, df)}
        if kind == "ssr_chi2":
            s = n * wald / df
            return {"statistic": s, "pvalue": float(stats.chi2.sf(s, q)), "df": q}
        if kind == "lr":
            s = n * np.log1p(wald / df)
            return {"statistic": s, "pvalue": float(stats.chi2.sf(s, q)), "df": q}
        raise ValueError(f"kind {kind!r} not recognised")

    # ------------------------------------------- impulse responses and FEVD
    # Reads of the fit, not state: ``O(H p k^3)`` for ``H`` periods, nothing kept.
    def coefs(self) -> np.ndarray:
        """``(p, k, k)``: ``A_1, ..., A_p`` (statsmodels' ``coefs``)."""
        b = self.slopes()
        return np.stack([b[j * self.k : (j + 1) * self.k].T for j in range(self.p)])

    def ma_rep(self, periods: int) -> np.ndarray:
        """``Phi_0 = I``, ``Phi_i = sum_{j=1}^{min(i, p)} Phi_{i-j} A_j``.

        ``(periods + 1, k, k)``, as ``VARResults.ma_rep`` and ``irf().irfs``.
        """
        a = self.coefs()
        phi = np.zeros((periods + 1, self.k, self.k))
        phi[0] = np.eye(self.k)
        for i in range(1, periods + 1):
            for j in range(1, min(i, self.p) + 1):
                phi[i] += phi[i - j] @ a[j - 1]
        return phi

    def orth_ma_rep(self, periods: int) -> np.ndarray:
        """``Theta_i = Phi_i P``, ``P`` the lower Cholesky factor of ``Sigma_u``."""
        return self.ma_rep(periods) @ np.linalg.cholesky(self.sigma_u())

    def fevd(self, periods: int) -> np.ndarray:
        """``(k, periods, k)``: equation x horizon x shock, as ``VARResults.fevd().decomp``.

        ``omega_{e j}(h) = sum_{i<h} Theta_i[e, j]^2 / MSE_h[e, e]``,
        ``MSE_h = sum_{i<h} Phi_i Sigma_u Phi_i'``.
        """
        theta2 = np.cumsum(self.orth_ma_rep(periods)[:periods] ** 2, axis=0)
        phi = self.ma_rep(periods)[:periods]
        mse = np.cumsum(np.einsum("hij,jl,hml->him", phi, self.sigma_u(), phi), axis=0)
        diag = np.diagonal(mse, axis1=1, axis2=2)  # (periods, k)
        return (theta2 / diag[:, :, None]).swapaxes(0, 1)

    def irf_cov(self, periods: int) -> np.ndarray:
        """Asymptotic covariance of ``vec(Phi_i)``, ``G_i Sigma_alpha G_i'`` (Lutkepohl 3.7.5).

        ``G_i = sum_{m<i} J (A')^{i-1-m} kron Phi_m`` with ``A`` the companion matrix
        and ``J`` its first ``k`` rows; ``Sigma_alpha`` is ``cov_params`` without the
        constant. ``(periods + 1, k^2, k^2)``, the first all zero.
        """
        k, m = self.k, self.m
        comp = np.zeros((m, m))
        comp[:k] = np.hstack(list(self.coefs()))
        comp[k:, :-k] = np.eye(m - k)
        phi = self.ma_rep(periods)
        cov_a = self.cov_params()[k:, k:]
        powers = [np.eye(m)]
        for _ in range(periods):
            powers.append(powers[-1] @ comp.T)
        out = np.zeros((periods + 1, k * k, k * k))
        for i in range(1, periods + 1):
            g = sum(np.kron(powers[i - 1 - j][:k], phi[j]) for j in range(i))
            out[i] = g @ cov_a @ g.T
        return out


@dataclass
class WindowVAR(OnlineVAR):
    """The VAR(p) on the last ``window`` rows of the stream only (no decay).

    A target counts while it and its ``p`` lags are all inside the window, so the
    fit is ``VAR(y[t - window + 1 : t + 1]).fit(p)``: ``window - p`` targets. Each
    row is added as above, then the target that has just lost its oldest lag is
    taken out by the reverse (down-dating) update, with ``W' = W - w``:

        e = x - xbar,  xbar' = xbar - (w / W') e
        Sxx' = Sxx - w (W / W') e e'        (and Sxy, Syy alike)

    The stored rows are the window itself (``window x k`` values and ``window``
    weights); the lag buffer is its last ``p`` rows, so the state is
    ``8 (window k + window + m + k + m^2 + m k + k^2 + 4)`` bytes (one counter more).
    """

    window: int = 0
    ys: np.ndarray = field(init=False)  # (window, k), oldest first
    ws: np.ndarray = field(init=False)
    n_pos: int = field(init=False, default=0)  # targets with w > 0 in the window

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.window <= self.p:
            raise ValueError("window must exceed p")
        if self.lam != 1.0:
            raise ValueError("this study's window has no decay")
        self.ys = np.full((self.window, self.k), np.nan)
        self.ws = np.zeros(self.window)

    def state_bytes(self) -> int:
        m, k, n = self.m, self.k, self.window
        return 8 * (n * k + n + m + k + m * m + m * k + k * k + 4)

    def step(self, y: np.ndarray, w: float = 1.0) -> tuple[np.ndarray, float]:
        y = np.asarray(y, dtype=float)
        pred, weight_before = super().step(y, w)  # adds the row, updates ``lags``
        if np.all(np.isfinite(y)) and self.n_seen > self.p and w > 0.0:
            self.n_pos += 1
        # The target whose oldest lag leaves now: ys[p] with lags ys[p-1], ..., ys[0].
        old = self.ys[: self.p + 1]
        if np.all(np.isfinite(old)):
            self._unlearn(old[-2::-1].reshape(-1), old[-1], float(self.ws[self.p]))
        self.ys[:-1] = self.ys[1:].copy()
        self.ys[-1] = y
        self.ws[:-1] = self.ws[1:].copy()
        self.ws[-1] = w
        return pred, weight_before

    def _unlearn(self, x: np.ndarray, y: np.ndarray, w: float) -> None:
        if w == 0.0:
            return
        self.n_pos -= 1
        if self.n_pos == 0:  # empty of weight: reset exactly, no 0/0 and no residue
            self.w_sum = self.w2_sum = 0.0
            for name in ("xbar", "ybar", "sxx", "sxy", "syy"):
                getattr(self, name)[...] = 0.0
            return
        w_old = self.w_sum
        w_new = w_old - w
        self.w2_sum -= w * w
        ex, ey = x - self.xbar, y - self.ybar
        self.xbar = self.xbar - (w / w_new) * ex
        self.ybar = self.ybar - (w / w_new) * ey
        g = w * w_old / w_new
        self.sxx = self.sxx - g * np.outer(ex, ex)
        self.sxy = self.sxy - g * np.outer(ex, ey)
        self.syy = self.syy - g * np.outer(ey, ey)
        self.w_sum = w_new


def run(
    y: np.ndarray,
    p: int,
    lam: float = 1.0,
    weights: np.ndarray | None = None,
    chunks: int = 1,
    min_weight: float = 0.0,
) -> tuple[OnlineVAR, np.ndarray, np.ndarray]:
    """Feed ``y`` (T x k) as ``chunks`` contiguous chunks; return model, preds, weight sums."""
    t, k = y.shape
    model = OnlineVAR(k, p, lam=lam, min_weight=min_weight)
    w = np.ones(t) if weights is None else np.asarray(weights, dtype=float)
    preds = np.empty((t, k))
    wsum = np.empty(t)
    for idx in np.array_split(np.arange(t), chunks):
        for i in idx:
            preds[i], wsum[i] = model.step(y[i], w[i])
    return model, preds, wsum
