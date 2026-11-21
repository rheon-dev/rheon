"""Size of the online Granger F test under decay, and its power after a break.

Under the null (``1`` does not Granger-cause ``0``), the share of streams whose
test, read at the end of the stream, rejects at 5%. With ``lam = 1`` this is the
textbook test; with ``lam < 1`` the degrees of freedom come from Kish's
``n_eff = W^2 / V``, an approximation this measures. Then a stream whose
``1 -> 0`` lag coefficient switches on half-way: the row at which the decayed
test first rejects at 1% and stays rejecting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from online_var import OnlineVAR, run  # noqa: E402


def var1(t: int, rng: np.random.Generator, b01: np.ndarray) -> np.ndarray:
    """Bivariate VAR(1); ``b01[i]`` is the lag-1 effect of variable 1 on 0 at row i."""
    y = np.zeros((t, 2))
    for i in range(1, t):
        a = np.array([[0.5, b01[i]], [0.3, 0.4]])
        y[i] = a @ y[i - 1] + rng.normal(size=2)
    return y


def size(lam: float, t: int, reps: int, seed: int = 0) -> tuple[float, float, float]:
    """Rejection rates of ``f`` and ``ssr_f`` at 5%, and the mean ``Sigma_u`` diagonal (truth 1)."""
    rng = np.random.default_rng(seed)
    rej_f, rej_w, sig = 0, 0, 0.0
    for _ in range(reps):
        m, _, _ = run(var1(t, rng, np.zeros(t)), 1, lam=lam)
        rej_f += m.test_causality(0, 1, kind="f")["pvalue"] < 0.05
        rej_w += m.test_causality(0, 1, kind="ssr_f")["pvalue"] < 0.05
        sig += float(np.mean(np.diag(m.sigma_u())))
    return rej_f / reps, rej_w / reps, sig / reps


def detection(lam: float, t: int = 2000, seed: int = 1) -> int | None:
    rng = np.random.default_rng(seed)
    b = np.where(np.arange(t) < t // 2, 0.0, 0.25)
    y = var1(t, rng, b)
    m = OnlineVAR(2, 1, lam=lam)
    pv = np.ones(t)
    for i in range(t):
        m.step(y[i])
        if i > 10:
            pv[i] = m.test_causality(0, 1, kind="ssr_f")["pvalue"]
    after = np.where(pv[t // 2 :] < 0.01)[0]
    for j in after:
        if np.all(pv[t // 2 + j :] < 0.01):
            return int(j)
    return None


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    for lam, t in ((1.0, 500), (0.99, 2000), (0.95, 2000)):
        f, s, sig = size(lam, t, reps)
        n_eff = (1 + lam) / (1 - lam) if lam < 1 else t - 1
        print(
            f"size at 5%: lam={lam} T={t} (n_eff~{n_eff:.0f}), {reps} reps:"
            f" f={f:.3f} ssr_f={s:.3f} mean sigma_u diag={sig:.3f}"
        )
    for lam in (1.0, 0.995, 0.99, 0.98):
        rows = detection(lam)
        print(
            f"break at row 1000 (b: 0 -> 0.25): lam={lam} rejects at 1% for good after {rows} rows"
        )
