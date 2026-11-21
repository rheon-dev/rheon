"""GARCH(1,1) online: the variance filter and a recursive quasi-maximum-likelihood
estimator, in numpy, one row at a time (docs/records/ORACLE-STUDY-485.md).

The model, on a zero-mean return ``r_t``::

    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}
    l_t      = -1/2 * (ln(2 pi) + ln sigma2_t + r_t^2 / sigma2_t)

``sigma2_t`` is computed before row ``t``'s return is read, so it is an
out-of-sample prediction of ``r_t^2``.

The start follows ``arch``'s default ("backcast" form (b) of the scan): the
first ``W = 75`` rows are buffered, the start value is their 0.94-weighted mean
of ``r^2``, and the buffered rows are then replayed through the recursion as
ordinary rows, with ``r_{-1}^2`` and ``sigma2_{-1}`` both read as the start
value. Those ``W`` rows emit nothing. With fixed parameters this is the path
``arch`` computes, which ``test_garch.py`` holds to the last bits.

The estimator (recursive prediction error on the Gaussian quasi-likelihood,
a stochastic Gauss-Newton step) carries the sensitivity
``d_t = d sigma2_t / d theta`` for ``theta = (omega, alpha, beta)`` by its own
recursion, at the current parameters::

    d_t   = (1, r_{t-1}^2, sigma2_{t-1}) + beta * d_{t-1}
    g_t   = 1/2 * (r_t^2 / sigma2_t - 1) * d_t / sigma2_t          (score of l_t)
    I_t   = 1/2 * d_t d_t' / sigma2_t^2                           (its information)
    H_t   = H_{t-1} + gamma_t * (I_t - H_{t-1})
    theta = Proj(theta + gamma_t * H_t^{-1} g_t),   gamma_t = 1 / (t + t0)

``Proj`` is the Euclidean projection onto ``omega >= omega_floor``,
``alpha, beta >= 0``, ``alpha + beta <= 1 - margin``; ``omega_floor`` is a
fixed fraction of the running mean of ``r^2``, so the step is scale-equivariant
(scaling ``r`` by ``c`` scales ``omega`` and every variance by ``c^2`` and
leaves ``alpha`` and ``beta`` alone). During the buffered replay ``H`` is
accumulated but ``theta`` is held, so the first step is taken with an
information matrix built on ``W`` rows.

State per target: ``theta`` (3), ``H`` (6 distinct), ``d`` (3), ``sigma2``,
the last ``r^2``, the running mean of ``r^2`` and its count, and the row
count: 16 floats, 128 bytes, plus the ``W``-row buffer until it is replayed.

numpy only. ``arch`` is the oracle and is imported only by ``compare`` and
the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LN_2PI = math.log(2.0 * math.pi)
BACKCAST_ROWS = 75
BACKCAST_DECAY = 0.94


def simulate(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    seed: int,
    burn: int = 1000,
    nu: float | None = None,
) -> np.ndarray:
    """A zero-mean Gaussian GARCH(1,1) series of ``n`` rows, after ``burn``
    discarded rows started at the unconditional variance. With ``nu``, the
    noise is Student-t with ``nu`` degrees of freedom scaled to unit variance."""
    rng = np.random.default_rng(seed)
    z = (
        rng.standard_normal(n + burn)
        if nu is None
        else (rng.standard_t(nu, n + burn) * math.sqrt((nu - 2.0) / nu))
    )
    r = np.empty(n + burn)
    s2 = omega / (1.0 - alpha - beta)
    prev_r2 = s2
    for t in range(n + burn):
        s2 = omega + alpha * prev_r2 + beta * s2
        r[t] = math.sqrt(s2) * z[t]
        prev_r2 = r[t] * r[t]
    return r[burn:]


def backcast(r2_head: np.ndarray) -> float:
    """``arch``'s default start: the 0.94-weighted mean of the first
    ``min(75, n)`` squared returns, weights normalized to one."""
    w = BACKCAST_DECAY ** np.arange(r2_head.shape[0])
    w = w / sum(w)
    return float(np.sum(r2_head * w))


def filter_variance(theta: np.ndarray, r: np.ndarray, start: float) -> np.ndarray:
    """The fixed-parameter filter, row by row: ``sigma2_t`` for every row."""
    omega, alpha, beta = (float(x) for x in theta)
    out = np.empty(r.shape[0])
    prev_r2 = start
    s2 = start
    for t in range(r.shape[0]):
        s2 = omega + alpha * prev_r2 + beta * s2
        out[t] = s2
        prev_r2 = float(r[t]) * float(r[t])
    return out


def loglik(r: np.ndarray, sigma2: np.ndarray) -> float:
    """The Gaussian log-likelihood summed over rows."""
    return float(-0.5 * np.sum(LN_2PI + np.log(sigma2) + r * r / sigma2))


def project(theta: np.ndarray, omega_floor: float, margin: float) -> np.ndarray:
    """Euclidean projection onto the admissible set (see the module docstring)."""
    omega = max(float(theta[0]), omega_floor)
    a, b = float(theta[1]), float(theta[2])
    cap = 1.0 - margin
    a, b = max(a, 0.0), max(b, 0.0)
    if a + b > cap:
        # project onto the line a + b = cap, then onto its segment in the quadrant
        shift = 0.5 * (a + b - cap)
        a, b = a - shift, b - shift
        if a < 0.0:
            a, b = 0.0, cap
        elif b < 0.0:
            a, b = cap, 0.0
    return np.array([omega, a, b])


@dataclass
class OnlineGarch:
    """The recursive quasi-likelihood estimator. ``update(r)`` takes one row.

    The defaults are the plain estimator of the module docstring. The options
    are the mitigations measured in the record's part 3; each default leaves the
    plain estimator's arithmetic unchanged:

    - ``target``: ``"running"`` sets ``omega = v * (1 - alpha - beta)`` after
      every step, ``v`` the running mean of ``r^2`` over every row read so far,
      and steps ``(alpha, beta)`` only; ``"warmup"`` takes ``v`` as the mean of
      ``r^2`` over the buffered rows, fixed from then on. The sensitivity of the
      two free parameters is ``J d`` with ``J = [[-v, 1, 0], [-v, 0, 1]]``
      (the chain rule through ``omega``), so no new recursion is carried.
    - ``step_cap``: each free parameter moves by at most ``step_cap`` times its
      own magnitude per step (0: no cap); the projection still follows.
    - ``gain``: the parameter step's gain, ``"inv"`` ``1/(t + t0)``, ``"sqrt"``
      ``1/sqrt(t + t0)``, ``"floor"`` ``max(1/(t + t0), gain_floor)``. The
      information matrix is averaged at ``1/(t + t0)`` under every gain.
    - ``batch_start``: the buffered rows (``warm`` of them) are fitted by
      ``arch``'s batch fit, standing in for a batch fit the model would own,
      and the parameters start there; the replay then builds the information
      matrix at those parameters. With ``target`` set (request A11), the batch
      fit is the targeted one (``vt_garch.vt_fit`` on the buffered rows) and
      ``omega`` starts at ``v_W (1 - alpha - beta)``, ``v_W`` the mean of the
      buffered rows' ``r^2``, which is that fit's own target.
    - ``gain="offset"`` (request A11, remedy (b), a slower early gain): the
      parameter step's gain is ``1/(t + gain_t0)``, smaller than
      ``1/(t + t0)`` early and the same order late; the information matrix
      keeps its ``1/(t + t0)`` average.
    """

    alpha0: float = 0.05
    beta0: float = 0.90
    t0: float = float(BACKCAST_ROWS)
    margin: float = 1e-4
    omega_floor_frac: float = 1e-6
    warm: int = BACKCAST_ROWS
    target: str = ""
    step_cap: float = 0.0
    gain: str = "inv"
    gain_floor: float = 1e-3
    gain_t0: float = 0.0
    batch_start: bool = False

    theta: np.ndarray = field(init=False)
    H: np.ndarray = field(init=False)
    d: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sigma2: float = math.nan
    mean_r2: float = 0.0
    vbar: float = math.nan
    n_seen: int = 0
    n_steps: int = 0
    buffer: list[float] = field(default_factory=list)
    started: bool = False

    # what the run emits, row by row (NaN on the buffered rows)
    pred: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.target in ("", "running", "warmup"), self.target
        assert self.gain in ("inv", "sqrt", "floor", "offset"), self.gain
        self.theta = np.array([math.nan, self.alpha0, self.beta0])
        k = 2 if self.target else 3
        self.H = np.zeros((k, k))

    def update(self, r: float) -> None:
        if not self.started:
            self.buffer.append(r)
            self.pred.append(math.nan)
            if len(self.buffer) == self.warm:
                self._start()
            return
        self.pred.append(self.sigma2)
        self._row(r, step=True)

    def _start(self) -> None:
        rows = np.asarray(self.buffer)
        b = backcast(rows[:BACKCAST_ROWS] ** 2)
        if self.batch_start and self.target:
            import vt_garch  # the targeted batch fit (lazy: vt_garch imports this module)

            _, res = vt_garch.vt_fit(rows)
            a, bb = (float(x) for x in np.asarray(res.params, dtype=float)[:2])
            self.theta = np.array([math.nan, a, bb])
        elif self.batch_start:
            _, res = arch_fit(rows)
            self.theta = np.asarray(res.params, dtype=float).copy()
        else:
            # omega from the start value at the initial persistence: theta0 targets b
            self.theta[0] = b * (1.0 - self.alpha0 - self.beta0)
        if self.target == "warmup":
            self.vbar = float(np.mean(rows * rows))
        if self.target:
            v = self.vbar if self.target == "warmup" else b
            if self.batch_start:
                v = float(np.mean(rows * rows))
            self.theta[0] = v * (1.0 - self.theta[1] - self.theta[2])
        self.started = True
        self.sigma2, self.d = self._advance(b, b, np.zeros(3))
        for x in rows:
            self._row(float(x), step=False)
        self.buffer = []

    def _advance(
        self, prev_r2: float, prev_s2: float, prev_d: np.ndarray
    ) -> tuple[float, np.ndarray]:
        omega, alpha, beta = self.theta
        s2 = omega + alpha * prev_r2 + beta * prev_s2
        d = np.array([1.0, prev_r2, prev_s2]) + beta * prev_d
        return float(s2), d

    def _v(self) -> float:
        return self.mean_r2 if self.target == "running" else self.vbar

    def _gain(self) -> float:
        t = self.n_steps + self.t0
        if self.gain == "sqrt":
            return 1.0 / math.sqrt(t)
        if self.gain == "floor":
            return max(1.0 / t, self.gain_floor)
        if self.gain == "offset":
            return 1.0 / (self.n_steps + self.gain_t0)
        return 1.0 / t

    def _row(self, r: float, step: bool) -> None:
        r2 = r * r
        self.n_seen += 1
        self.mean_r2 += (r2 - self.mean_r2) / self.n_seen
        s2, d = self.sigma2, self.d
        if self.target:
            v = self._v()
            dfree = np.array([d[1] - v * d[0], d[2] - v * d[0]])
        else:
            dfree = d
        info = 0.5 * np.outer(dfree, dfree) / (s2 * s2)
        if step:
            self.n_steps += 1
            gamma = 1.0 / (self.n_steps + self.t0)
            self.H += gamma * (info - self.H)
            g = 0.5 * (r2 / s2 - 1.0) * dfree / s2
            move = self._gain() * np.linalg.solve(self.H, g)
            free = self.theta[1:] if self.target else self.theta
            if self.step_cap > 0.0:
                lim = self.step_cap * np.abs(free)
                move = np.clip(move, -lim, lim)
            proposal = self.theta.copy()
            if self.target:
                proposal[1:] += move
            else:
                proposal += move
            self.theta = project(proposal, self.omega_floor_frac * self.mean_r2, self.margin)
            if self.target:
                self.theta[0] = self._v() * (1.0 - self.theta[1] - self.theta[2])
        else:
            # the buffered replay: an equal-weight mean of the information
            self.H += (info - self.H) / self.n_seen
        self.sigma2, self.d = self._advance(r2, s2, d)


def run(r: np.ndarray, **kw: object) -> OnlineGarch:
    """Feed a series one row at a time."""
    est = OnlineGarch(**kw)  # type: ignore[arg-type]
    for x in r:
        est.update(float(x))
    return est


def arch_fit(r: np.ndarray):  # noqa: ANN201
    """The oracle: ``arch``'s batch fit, zero mean, GARCH(1,1), normal,
    no rescaling, its default backcast and bounds."""
    from arch import arch_model

    am = arch_model(r, mean="Zero", vol="GARCH", p=1, o=0, q=1, dist="normal", rescale=False)
    return am, am.fit(disp="off", cov_type="robust")


def compare(n: int, seed: int, truth: tuple[float, float, float], **kw: object) -> dict[str, float]:
    """The online estimator (options ``kw``) against ``arch`` on one simulated series."""
    r = simulate(n, *truth, seed=seed)
    am, res = arch_fit(r)
    theta_b = np.asarray(res.params, dtype=float)
    se_b = np.asarray(res.std_err, dtype=float)
    est = run(r, **kw)
    theta_o = est.theta.copy()

    start = backcast(r[:BACKCAST_ROWS] ** 2)
    path_b = filter_variance(theta_b, r, start)
    path_o = filter_variance(theta_o, r, start)
    llf_b = float(res.loglikelihood)
    llf_o = float(am.fix(theta_o).loglikelihood)

    # the online run's own one-step predictions, against the batch-parameter path
    pred = np.asarray(est.pred)
    emitted = np.isfinite(pred)
    tail = slice(n // 2, n)
    out = {
        "n": float(n),
        "seed": float(seed),
        "arch_converged": float(res.convergence_flag),  # 0: SLSQP converged
        "arch_path_vs_filter_max_rel": float(
            np.max(np.abs(path_b / res.conditional_volatility**2 - 1.0))
        ),
        "own_llf_vs_arch_abs": abs(loglik(r, path_b) - llf_b),
        "llf_batch": llf_b,
        "llf_online_params": llf_o,
        "llf_diff": llf_b - llf_o,
        "llf_diff_per_row": (llf_b - llf_o) / n,
        # the same rows (the buffered ones emit nothing): the online run's
        # out-of-sample predictions against the batch fit's in-sample path
        "online_prequential_llf": loglik(r[emitted], pred[emitted]),
        "batch_llf_same_rows": loglik(r[emitted], path_b[emitted]),
        "path_median_rel_final_params": float(np.median(np.abs(path_o / path_b - 1.0))),
        "pred_median_rel_second_half": float(np.median(np.abs(pred[tail] / path_b[tail] - 1.0))),
    }
    for i, name in enumerate(("omega", "alpha", "beta")):
        out[f"{name}_true"] = truth[i]
        out[f"{name}_batch"] = float(theta_b[i])
        out[f"{name}_online"] = float(theta_o[i])
        out[f"{name}_diff"] = float(theta_o[i] - theta_b[i])
        out[f"{name}_batch_se"] = float(se_b[i])
        out[f"{name}_diff_in_se"] = float((theta_o[i] - theta_b[i]) / se_b[i])
    return out


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 485
    for k, v in compare(n, seed, (0.02, 0.08, 0.90)).items():
        print(f"{k:34s} {v: .10g}")
