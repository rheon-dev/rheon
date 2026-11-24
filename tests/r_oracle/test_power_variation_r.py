"""R's ``yuima`` (CRAN 1.15.34, GPL-2), ``highfrequency`` (1.0.3, GPL >= 2)
and ``sandwich`` (3.1-0, GPL-2 | GPL-3) as the oracles of
``power_variation``'s terms (docs/PLAN.md task 555), through rpy2.
Selected by ``-m r_oracle`` alone, which the ``r-oracles`` job of
``.github/workflows/oracles.yml`` runs with R and its packages installed.

Their conventions, read in their source and measured, stated here so the
comparison is of the arithmetic:

- ``yuima::mpv(r = c(p_1, ..., p_m), normalize = TRUE)`` is
  ``n^(Σp/2 - 1) Π μ_{p_i}^-1 Σ Π |Δ|^{p_i}``, ``r[1]`` on the oldest
  difference, ``n`` the differences: the normalized multipower term, any
  powers, the one library that takes unequal ones.
- ``highfrequency::rMPVar(m, p)`` is the same with ``m`` equal powers
  ``p/m`` and a factor ``N / (N - m + 1)``, divided out here.
- ``highfrequency::rKernelCov(kernelType, kernelParam = H, kernelDOFadj =
  FALSE)`` is the flat-top realized kernel, ``γ_0 + Σ_{h=1}^{H} k((h -
  1)/H)(γ_h + γ_-h)``, uncentered and undivided; ``kernelDOFadj = TRUE``
  would multiply ``γ_h`` by ``n/(n - h)``, which the term does not.
- ``sandwich::kernHAC(lm(x ~ 1), bw, kernel, prewhite = FALSE, adjust =
  FALSE, sandwich = FALSE)`` is the HAC long-run variance about the mean,
  divided by ``n``, at weights ``k(h / bw)``: ``bw = H + 1`` is the term's
  ``k(h / (H + 1))``, and ``n`` times it is the ``center`` kernel term.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from rheon import polars as rp

TIER = "essential"

pytestmark = pytest.mark.r_oracle

yuima = importr("yuima")
hf = importr("highfrequency")
sandwich = importr("sandwich")
stats = importr("stats")

INF = float("inf")


def bounced(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = 0.0005 * np.exp(np.cumsum(0.02 * rng.standard_normal(n)))
    noise = 0.0003 * rng.standard_normal(n + 1)
    return vol * rng.standard_normal(n) + np.diff(noise)


def _last(terms: dict, r: np.ndarray) -> dict:
    df = pl.DataFrame({"r": [*map(float, r), None]})
    spec = rp.spec.power_variation("v", targets=["r"], terms=terms, half_life=INF)
    return rp.ModelBank([spec]).fit_predict(df).unnest("v").row(-1, named=True)


def _ulps(got: float, want: float) -> float:
    return abs(got - want) / np.spacing(abs(want))


def _float(x) -> float:
    return float(np.asarray(x).ravel()[0])


POWERS = [
    [2.0],
    [1.0, 1.0],
    [2 / 3, 2 / 3, 2 / 3],
    [4 / 3, 4 / 3, 4 / 3],
    [1.0, 0.5],
    [0.5, 1.5, 1.0],
    [2.0, 2.0],
]


@pytest.mark.parametrize("powers", POWERS, ids=lambda p: "-".join(f"{x:.3g}" for x in p))
@pytest.mark.parametrize("n", [390, 4_680])
def test_normalized_multipower_terms_are_yuimas_mpv(powers, n):
    """Each set of powers, unequal and in either order, against ``mpv`` on
    the prices whose differences the returns are. Measured: at most 9 ulps
    (the non-integer powers and ``μ_p``, which each side computes its own
    way)."""
    r = bounced(n, n + len(powers))
    y = np.concatenate([[0.0], np.cumsum(r)])
    data = yuima.setData(ro.FloatVector(y), delta=1.0 / n)
    want = _float(yuima.mpv(data, r=ro.FloatVector(powers), normalize=True))
    # The differences of the prices R reads, so the two sides take the same
    # returns to the bit.
    got = _last({"t": {"powers": powers, "normalize": True}}, np.diff(y))["t_r"]
    assert _ulps(got, want) <= 16, (got, want)


@pytest.mark.parametrize(("m", "p"), [(2, 2.0), (2, 3.0), (3, 2.0), (3, 4.0), (4, 4.0)])
def test_equal_powers_are_highfrequencys_rmpvar(m, p):
    """``rMPVar(m, p)``: ``m`` powers of ``p/m``, its ``N / (N - m + 1)``
    divided out. Measured: at most 9 ulps."""
    r = bounced(2_340, 3 * m)
    want = _float(hf.rMPVar(ro.r.matrix(ro.FloatVector(r), ncol=1), m=m, p=p))
    want *= (len(r) - m + 1) / len(r)
    got = _last({"t": {"powers": [p / m] * m, "normalize": True}}, r)["t_r"]
    assert _ulps(got, want) <= 16, (got, want)


KERNELS = {
    "bartlett": "Bartlett",
    "parzen": "Parzen",
    "tukey_hanning": "TukeyHanning",
    "rectangular": "Rectangular",
}


@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("h", [1, 4, 20])
def test_flat_top_kernel_terms_are_highfrequencys_rkernelcov(kernel, h):
    """``rKernelCov`` without its degrees-of-freedom factor is the flat-top
    term, each kernel. Measured: at most 2.2 ε of the scale ``(2H + 1) Σ
    r²``, the two summing the lags in different orders."""
    r = bounced(2_340, 7 * h)
    want = _float(
        hf.rKernelCov(
            ro.FloatVector(r), kernelType=KERNELS[kernel], kernelParam=h, kernelDOFadj=False
        )
    )
    got = _last({"k": {"kernel": kernel, "bandwidth": h, "flat_top": True}}, r)["k_r"]
    scale = (2 * h + 1) * float(np.dot(r, r))
    assert abs(got - want) <= 8 * np.finfo(float).eps * scale, (got, want)


SANDWICH = {"bartlett": "Bartlett", "parzen": "Parzen", "tukey_hanning": "Tukey-Hanning"}


@pytest.mark.parametrize("kernel", sorted(SANDWICH))
@pytest.mark.parametrize("h", [1, 5, 20])
def test_centered_kernel_terms_are_sandwichs_hac_times_n(kernel, h):
    """``kernHAC`` on an intercept-only regression, no prewhitening and no
    small-sample adjustment, at ``bw = H + 1``: the long-run variance about
    the mean over ``n``. Measured: at most 1.2 ε of the scale."""
    r = bounced(3_000, 11 * h)
    ro.globalenv["x"] = ro.FloatVector(r)
    fit = stats.lm(ro.Formula("x ~ 1"))
    want = _float(
        sandwich.kernHAC(
            fit,
            bw=h + 1.0,
            kernel=SANDWICH[kernel],
            prewhite=False,
            adjust=False,
            sandwich=False,
        )
    ) * len(r)
    got = _last({"k": {"kernel": kernel, "bandwidth": h, "center": True}}, r)["k_r"]
    scale = (2 * h + 1) * float(np.dot(r, r))
    assert abs(got - want) <= 8 * np.finfo(float).eps * scale, (got, want)
