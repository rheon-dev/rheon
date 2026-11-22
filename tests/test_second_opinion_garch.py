"""`garch` against `arch` (task 492): the online estimator, through the bank,
against `arch`'s batch fit of the same rows.

Each simulated series is fitted twice: by the bank, one row at a time, and
by `arch`'s maximum likelihood on all its rows. Four quantities compare
them, each a seed's:

- **the gap** of each parameter, in `arch`'s (robust) standard errors, and
  whether every one is within 2;
- **D**, the likelihood-ratio gap at the last row, `l_n(θ_arch) -
  l_n(θ_online)`, both by `arch`'s own likelihood (`am.fix`). `2D` is the
  squared distance of the online estimate from the batch fit in the batch
  fit's own metric, whose sampling error has `E[2D] = k`;
- **the out-of-sample excess**, the log-likelihood `arch`'s in-sample path
  gains over the bank's out-of-sample variances on the rows after the start,
  against its expected size `T = (k/2)(ln(n/n0) + 1)` for `k` parameters
  fitted by likelihood after a start of `n0` rows (the prequential `(k/2) ln
  n` and the batch fit's in-sample optimism);
- at 100,000 rows, for the forms with a free `omega`, D and the gap again:
  the recursive estimate agrees with the batch fit to first order, so both
  fall as the rows grow. The ratio of the median gap at 100,000 rows to
  that at 10,000 is reported, not held: its spread over 50 seeds is 0.06 to
  0.13, and after a 2,000-row start its expected value lies between 0.49
  (the martingale term) and 0.77 (the bias term).

The bounds, each from a 50-seed grid of the build (study 492, the review
of 2026-11-21): count floors four seeds under the measurement; the excess's
median at most 1.5 T and every seed's at most 4 T; D's 90th percentile at
most `chi2_k(0.95)/2`, the batch fit's own 95% region; at 100,000 rows the
median D at most `k/32` and its 90th percentile at most `k/8` (a quarter and
a half of one batch sampling deviation), and the median gap at most 0.25 SE.

The variance-targeted form is another estimand, held to `arch`'s targeted
fit (`arch`'s own `GARCH` with `omega = v_T (1 - alpha - beta)`): the 2 SE
count, the median gap at most 0.5 SE (the Huber clip alone costs about
0.24), D's 90th percentile at most 3.0, and in place of the excess bound
the excess over that of an expanding-window targeted refit (the batch
targeted fit of each prefix, every 100 rows, its level the running mean
square, forecasting out of sample): median at most 1 nat, every seed at
most 6. It owes no shrinkage with the rows.

The t truths have `nu = 6`, so the returns' fourth moment is finite and the
standard errors exist. EGARCH is held at the study's truth, `beta = 0.97`,
and at `beta = 0.985`, where the stricter invertibility guards bind.

**The start itself** (PLAN 548, STUDY-549) is held to `arch` on its own
rows: its fit scored by `arch`'s likelihood (`am.fix`) against the best of
`arch`'s fits from its default start and from the start's fit, at starts of
500, 1,000 and 2,000 rows, on these truths and on persistence-0.90 ones
(`LO`). At 1,000 and 2,000 rows no start may be short of `arch` by more
than 1e-6 nats (measured 9.1e-13 at most, 1,350 starts); at 500 rows a
start can converge to a lower local maximum the grid's best point chose
(2 of 650 short by more than 0.01 nats, at most one a cell), held at two a
cell. A start waits until its likelihood-ratio statistic against constant
variance reaches chi-squared at 0.999 for its restrictions (2 for
`garch`, 3 for `gjr`, `egarch` and `figarch`); that wait is held to
`arch`'s own test, its GARCH fit against its `ConstantVariance` fit, and
withholds no 2,000-row start of these truths.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from arch import arch_model
from arch.univariate import FIGARCH, GARCH, Normal, ZeroMean
from arch.univariate import recursions as arch_rec
from arch.univariate.distribution import SkewStudent, StudentsT
from arch.univariate.recursions import figarch_weights
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import chi2

from rheon import polars as rp

TIER = "mixed"

LN_2PI = math.log(2.0 * math.pi)
N0 = 2000
#: (vol, dist, the truth, k)
FORMS = {
    "garch": ("garch", "normal", (0.05, 0.08, 0.90), 3),
    "gjr": ("gjr", "normal", (0.03, 0.05, 0.10, 0.87), 4),
    "egarch": ("egarch", "normal", (0.0, 0.15, -0.08, 0.97), 4),
    "egarch_985": ("egarch", "normal", (0.0, 0.15, -0.08, 0.985), 4),
    "garch_t": ("garch", "t", (0.05, 0.08, 0.90), 4),
    "garch_skewt": ("garch", "skewt", (0.05, 0.08, 0.90), 5),
    "gjr_t": ("gjr", "t", (0.03, 0.05, 0.10, 0.87), 5),
    "figarch": ("figarch", "normal", (0.03, 0.2, 0.4, 0.5), 4),
}
SHAPE = {"normal": (), "t": (6.0,), "skewt": (6.0, -0.2)}
#: The persistence-0.90 truths: (vol, dist, the truth).
LO = {
    "garch": ("garch", "normal", (0.10, 0.10, 0.80)),
    "gjr": ("gjr", "normal", (0.10, 0.05, 0.10, 0.80)),
    "egarch": ("egarch", "normal", (0.0, 0.20, -0.10, 0.90)),
    "garch_t": ("garch", "t", (0.10, 0.10, 0.80)),
    "garch_skewt": ("garch", "skewt", (0.10, 0.10, 0.80)),
    "gjr_t": ("gjr", "t", (0.10, 0.05, 0.10, 0.80)),
}
#: The likelihood-ratio statistic's cut, chi-squared at 0.999 for the
#: restrictions constant variance puts on each process.
LR_CUT = {"garch": float(chi2.ppf(0.999, 2)), "gjr": float(chi2.ppf(0.999, 3))}
LR_CUT["egarch"] = LR_CUT["figarch"] = LR_CUT["gjr"]
ABS_Z = math.sqrt(2.0 / math.pi)


def innovations(dist: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if dist == "normal":
        return rng.standard_normal(n)
    if dist == "t":
        return rng.standard_t(6.0, n) * math.sqrt(4.0 / 6.0)
    return np.asarray(SkewStudent(seed=rng).simulate(list(SHAPE["skewt"]))(n))


def simulate(form: str, n: int, seed: int) -> np.ndarray:
    """The form's process at its truth, after 1,000 rows discarded (3,000
    for FIGARCH, whose lags reach 1,000 back), its innovations seeded by the
    form and the seed."""
    vol, dist, th, _ = FORMS[form]
    rng = np.random.default_rng([492, sorted(FORMS).index(form), seed])
    burn = 3000 if vol == "figarch" else 1000
    z = innovations(dist, n + burn, rng)
    out = np.empty(n + burn)
    if vol == "figarch":
        lam = figarch_weights(np.array(th[1:]), 1, 1, 1000)[::-1]
        w = th[0] / (1.0 - th[3])
        x2 = np.full(1000 + n + burn, w / (1.0 - lam.sum()))
        for t in range(n + burn):
            out[t] = math.sqrt(w + float(lam @ x2[t : t + 1000])) * z[t]
            x2[1000 + t] = out[t] * out[t]
    elif vol == "egarch":
        om, a, g, b = th
        lv, zp = om / (1.0 - b), 0.0
        for t in range(n + burn):
            lv = om + a * (abs(zp) - ABS_Z) + g * zp + b * lv
            out[t] = math.exp(0.5 * lv) * z[t]
            zp = z[t]
    else:
        gjr = vol == "gjr"
        om, a, g, b = th if gjr else (th[0], th[1], 0.0, th[2])
        s = om / (1.0 - a - 0.5 * g - b)
        e = 0.0
        for t in range(n + burn):
            s = om + (a + (g if e < 0 else 0.0)) * e * e + b * s
            e = math.sqrt(s) * z[t]
            out[t] = e
    return out[burn:]


def simulate_lo(form: str, n: int, seed: int) -> np.ndarray:
    """A persistence-0.90 truth (`LO`), after 1,000 rows discarded, its
    innovations seeded by the form and the seed."""
    vol, dist, th = LO[form]
    rng = np.random.default_rng([549, sorted(LO).index(form), seed])
    z = innovations(dist, n + 1000, rng)
    out = np.empty(n + 1000)
    if vol == "egarch":
        om, a, g, b = th
        lv, zp = om / (1.0 - b), 0.0
        for t in range(n + 1000):
            lv = om + a * (abs(zp) - ABS_Z) + g * zp + b * lv
            out[t] = math.exp(0.5 * lv) * z[t]
            zp = z[t]
    else:
        om, a, g, b = th if vol == "gjr" else (th[0], th[1], 0.0, th[2])
        s, e = om / (1.0 - a - 0.5 * g - b), 0.0
        for t in range(n + 1000):
            s = om + (a + (g if e < 0 else 0.0)) * e * e + b * s
            e = math.sqrt(s) * z[t]
            out[t] = e
    return out[1000:]


def arch_model_of(form: str, r: np.ndarray):  # noqa: ANN201
    vol, dist, _, _ = FORMS[form]
    if vol == "figarch":
        return ZeroMean(
            r, volatility=FIGARCH(p=1, q=1, truncation=1000), distribution=Normal(), rescale=False
        )
    kw = {"garch": dict(vol="GARCH", p=1, o=0, q=1), "gjr": dict(vol="GARCH", p=1, o=1, q=1)}.get(
        vol, dict(vol="EGARCH", p=1, o=1, q=1)
    )
    return arch_model(r, mean="Zero", dist=dist, rescale=False, **kw)


def bank_run(form: str, r: np.ndarray, targeting: bool = False):  # noqa: ANN201
    """The bank's variances and its parameters before each row (those
    reported after the row before), and its parameters at the end."""
    vol, dist, _, _ = FORMS[form]
    spec = rp.spec.garch(
        "g", targets=["r"], vol=vol, dist=dist, variance_targeting=targeting, coef_every=0
    )
    out = rp.ModelBank([spec]).fit_predict(pl.DataFrame({"r": r})).unnest("g")
    pred = out["variance_r"].to_numpy().astype(float)
    coef = out["coef"].to_list()
    before = [None] + coef[:-1]
    return pred, before, np.asarray(coef[-1], dtype=float)


def row_loglik(dist: str, r: np.ndarray, s: np.ndarray, shape) -> np.ndarray:  # noqa: ANN001
    """Each row's log-likelihood under `arch`'s densities."""
    if dist == "normal":
        return -0.5 * (LN_2PI + np.log(s) + r * r / s)
    d = StudentsT() if dist == "t" else SkewStudent()
    return np.asarray(d.loglikelihood(np.asarray(shape, dtype=float), r, s, individual=True))


def compare(form: str, r: np.ndarray) -> dict:
    """One series' gap, D and out-of-sample excess against `arch`."""
    vol, dist, _, k = FORMS[form]
    am = arch_model_of(form, r)
    res = am.fit(disp="off", options={"maxiter": 1000})
    bp, se = np.asarray(res.params), np.asarray(res.std_err)
    path = np.asarray(res.conditional_volatility) ** 2
    pred, before, fin = bank_run(form, r)
    gap = (fin - bp) / se
    n0 = int(np.argmax(np.isfinite(pred)))
    m = len(SHAPE[dist])
    ka = len(bp) - m
    arch_rows = row_loglik(dist, r[n0:], path[n0:], bp[ka:])
    if m:
        online = np.array(
            [
                row_loglik(dist, r[t : t + 1], pred[t : t + 1], before[t][ka:])[0]
                for t in range(n0, len(r))
            ]
        )
    else:
        online = row_loglik(dist, r[n0:], pred[n0:], ())
    excess = float(np.sum(arch_rows - online))
    n = len(r)
    return {
        "gap": gap,
        "within2": bool(np.all(np.abs(gap) <= 2.0)),
        "D": float(res.loglikelihood - am.fix(fin).loglikelihood),
        "excess": excess,
        "T": (k / 2.0) * (math.log(n / n0) + 1.0),
        "k": k,
    }


# --- the essentials: two seeds a form ---------------------------------------


@pytest.mark.parametrize("form", sorted(FORMS))
def test_two_seeds_a_form_are_arch_to_first_order(form):
    """Each of two seeds at 10,000 rows: every parameter within 2.5 of
    `arch`'s standard errors, D inside the batch fit's 95% region, the
    out-of-sample excess within 4 T."""
    for seed in (0, 1):
        c = compare(form, simulate(form, 10_000, seed))
        assert np.max(np.abs(c["gap"])) <= 2.5, (form, seed, c["gap"])
        assert c["D"] <= chi2.ppf(0.95, c["k"]) / 2.0, (form, seed, c["D"])
        assert c["excess"] <= 4.0 * c["T"], (form, seed, c["excess"], c["T"])


# --- the extended grids: 50 seeds a form ------------------------------------

SEEDS = range(50)
#: Within 2 SE on every parameter: measured 50 of 50 on every form (both
#: EGARCH truths included); the floors are four seeds under.
FLOOR = 46


@pytest.mark.extended(reason="50 seeds a form at 10,000 rows, an arch fit each")
@pytest.mark.parametrize("form", sorted(FORMS))
def test_fifty_seeds_a_form_against_arch(form):
    cs = [compare(form, simulate(form, 10_000, s)) for s in SEEDS]
    k = cs[0]["k"]
    within = sum(c["within2"] for c in cs)
    ratio = np.array([c["excess"] / c["T"] for c in cs])
    d = np.array([c["D"] for c in cs])
    assert within >= FLOOR, (form, within)
    assert np.median(ratio) <= 1.5, (form, np.median(ratio))
    assert ratio.max() <= 4.0, (form, ratio.max())
    assert np.percentile(d, 90) <= chi2.ppf(0.95, k) / 2.0, (form, np.percentile(d, 90))


#: The forms with a free `omega` held at 100,000 rows.
FREE = sorted(f for f in FORMS if f != "egarch_985")


@pytest.mark.extended(reason="50 seeds a form at 100,000 rows, two arch fits each")
@pytest.mark.parametrize("form", FREE)
def test_at_100k_rows_the_free_forms_are_the_batch_fit(form):
    """At 100,000 rows: median D at most k/32, its 90th percentile at most
    k/8, the median gap at most 0.25 SE. The ratio of the median gap to its
    value at 10,000 rows (the first 10,000 of the same series) is printed."""
    big, small = [], []
    for s in SEEDS:
        r = simulate(form, 100_000, s)
        big.append(compare(form, r))
        small.append(compare(form, r[:10_000]))
    k = big[0]["k"]
    d = np.array([c["D"] for c in big])
    g = np.median(np.abs(np.array([c["gap"] for c in big])))
    g10 = np.median(np.abs(np.array([c["gap"] for c in small])))
    print(f"{form}: median |gap| {g10:.3f} at 10k, {g:.3f} at 100k, ratio {g / g10:.2f}")
    assert np.median(d) <= k / 32.0, (form, np.median(d))
    assert np.percentile(d, 90) <= k / 8.0, (form, np.percentile(d, 90))
    assert g <= 0.25, (form, g)


# --- variance targeting -------------------------------------------------------

VT_TRUTH = (0.02, 0.08, 0.90)


def vt_process():  # noqa: ANN201
    """`arch`'s GARCH(1,1) with `omega` targeted at the mean square of the
    rows it fits: two free parameters."""

    class VTGarch(GARCH):
        def __init__(self) -> None:
            super().__init__(p=1, o=0, q=1)
            self._num_params = 2
            self._name = "VT-GARCH"

        def compute_variance(self, parameters, resids, sigma2, backcast, var_bounds):  # noqa: ANN001, ANN201
            parameters, resids, backcast, var_bounds = self._check_compute_variance(
                parameters, resids, sigma2, backcast, var_bounds
            )
            a, b = float(parameters[0]), float(parameters[1])
            full = np.array([float(np.mean(resids**2)) * (1.0 - a - b), a, b])
            arch_rec.garch_recursion(
                full,
                resids**2,
                np.sign(resids),
                sigma2,
                1,
                0,
                1,
                resids.shape[0],
                backcast,
                var_bounds,
            )
            return sigma2

        def bounds(self, resids):  # noqa: ANN001, ANN201
            return [(0.0, 1.0), (0.0, 1.0)]

        def constraints(self):  # noqa: ANN201
            return np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]), np.array([0.0, 0.0, -1.0])

        def starting_values(self, resids):  # noqa: ANN001, ANN201
            return np.array([0.08, 0.90])

        def parameter_names(self) -> list[str]:
            return ["alpha[1]", "beta[1]"]

    return VTGarch()


def vt_simulate(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([492, 99, seed])
    om, a, b = VT_TRUTH
    s, e, out = om / (1 - a - b), 0.0, np.empty(n + 1000)
    for t, z in enumerate(rng.standard_normal(n + 1000)):
        s = om + a * e * e + b * s
        e = math.sqrt(s) * z
        out[t] = e
    return out[1000:]


def backcast(r2: np.ndarray) -> float:
    w = 0.94 ** np.arange(min(75, len(r2)))
    return float(np.sum(r2[: len(w)] * w) / w.sum())


def vt_path(om: float, a: float, b: float, r2: np.ndarray, b0: float) -> np.ndarray:
    x = np.empty_like(r2)
    x[0] = om + a * b0 + b * b0
    x[1:] = om + a * r2[:-1]
    return lfilter([1.0], [1.0, -b], x)


def vt_prefix_fit(r2: np.ndarray, b0: float, x0) -> tuple[float, float]:  # noqa: ANN001
    """The targeted likelihood of a prefix, maximized by scipy: the
    definition by hand, the optimizer the library's."""
    v = float(r2.mean())

    def f(p):  # noqa: ANN001, ANN202
        a, b = p
        if a < 0 or b < 0 or a + b >= 1:
            return 1e9
        s = vt_path(v * (1 - a - b), a, b, r2, b0)
        return 0.5 * float(np.mean(np.log(s) + r2 / s))

    best = None
    for st in (x0, (0.08, 0.90)):
        r = minimize(
            f, st, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-13, maxiter=2000)
        )
        if best is None or r.fun < best.fun:
            best = r
    return float(best.x[0]), float(best.x[1])


def vt_compare(seed: int, n: int = 10_000) -> dict:
    r = vt_simulate(n, seed)
    am = arch_model(r, mean="Zero", vol="GARCH", dist="normal", rescale=False)
    am.volatility = vt_process()
    res = am.fit(disp="off", cov_type="robust")
    ab, se = np.asarray(res.params)[:2], np.asarray(res.std_err)[:2]
    pred, _, fin = bank_run("garch", r, targeting=True)
    r2 = r * r
    b0 = backcast(r2)
    vt = float(r2.mean())
    path = vt_path(vt * (1 - ab.sum()), ab[0], ab[1], r2, b0)
    n0 = int(np.argmax(np.isfinite(pred)))

    def lrow(s: np.ndarray) -> np.ndarray:
        return -0.5 * (np.log(s) + r2[n0:] / s)

    excess = float(np.sum(lrow(path[n0:]) - lrow(pred[n0:])))
    # The expanding-window targeted refit: every 100 rows the batch targeted
    # fit of the prefix, the level the running mean square, out of sample.
    vbar = np.cumsum(r2) / np.arange(1, n + 1)
    ref = np.full(n, np.nan)
    cur, x0 = None, (0.08, 0.90)
    for c in range(n0, n, 100):
        a, b = vt_prefix_fit(r2[:c], b0, x0)
        x0 = (a, b)
        s = vt_path(r2[:c].mean() * (1 - a - b), a, b, r2[: c + 1], b0)[c] if cur is None else cur
        for t in range(c, min(c + 100, n)):
            if t > c or cur is not None:
                s = vbar[t - 1] * (1 - a - b) + a * r2[t - 1] + b * s
            ref[t] = s
        cur = s
    over_ref = float(np.sum(lrow(path[n0:]) - lrow(ref[n0:])))
    return {
        "gap": (fin[1:3] - ab) / se,
        "D": float(res.loglikelihood - am.fix(fin[1:3]).loglikelihood),
        "excess_over_refit": excess - over_ref,
    }


def test_the_targeted_form_is_arch_targeted_fit_on_two_seeds():
    for seed in (0, 1):
        c = vt_compare(seed)
        assert np.max(np.abs(c["gap"])) <= 2.0, (seed, c["gap"])
        assert c["D"] <= 3.0, (seed, c["D"])
        assert c["excess_over_refit"] <= 6.0, (seed, c["excess_over_refit"])


@pytest.mark.extended(reason="50 seeds at 10,000 rows, an arch fit and 80 prefix refits each")
def test_fifty_seeds_of_the_targeted_form():
    cs = [vt_compare(s) for s in SEEDS]
    within = sum(bool(np.all(np.abs(c["gap"]) <= 2.0)) for c in cs)
    gaps = np.abs(np.array([c["gap"] for c in cs]))
    d = np.array([c["D"] for c in cs])
    over = np.array([c["excess_over_refit"] for c in cs])
    assert within >= FLOOR, within
    assert np.median(gaps) <= 0.5, np.median(gaps)
    assert np.percentile(d, 90) <= 3.0, np.percentile(d, 90)
    assert np.median(over) <= 1.0, np.median(over)
    assert over.max() <= 6.0, over.max()


# --- the start against arch on its own rows (PLAN 548) -----------------------


def vol_dist(truth: str, form: str) -> tuple[str, str]:
    return FORMS[form][:2] if truth == "hi" else LO[form][:2]


def series_of(truth: str, form: str, n: int, seed: int) -> np.ndarray:
    return simulate(form, n, seed) if truth == "hi" else simulate_lo(form, n, seed)


def start_of(truth: str, form: str, r: np.ndarray, start: int) -> dict:
    """The bank's start on `r` with `start_rows=start`: the reason its first
    fit gave the row after it, the row it first reported on, and, where it
    reported, its fit against `arch`'s on the start's own rows -- `short`,
    the best of `arch`'s fits from its default start and from the start's
    fit, less the start's fit, both by `arch`'s likelihood."""
    vol, dist = vol_dist(truth, form)
    spec = rp.spec.garch("g", targets=["r"], vol=vol, dist=dist, start_rows=start, coef_every=0)
    out = rp.ModelBank([spec]).fit_predict(pl.DataFrame({"r": r})).unnest("g")
    var = out["variance_r"].to_numpy().astype(float)
    reason = out["withheld_reason"].cast(pl.String).to_list()
    got = {"first": reason[start], "n0": None, "short": None}
    if not np.isfinite(var).any():
        return got
    n0 = int(np.argmax(np.isfinite(var)))
    # a start's buffer doubles in place, or is buffered afresh at four times
    rows = r[:n0] if n0 in (start, 2 * start, 4 * start) else r[n0 - 4 * start : n0]
    theta = np.asarray(out["coef"].to_list()[n0 - 1], dtype=float)
    am = arch_model_of(form, rows) if truth == "hi" else lo_model(form, rows)
    ours = float(am.fix(theta).loglikelihood)
    best = max(
        float(am.fit(disp="off", starting_values=sv, options={"maxiter": 1000}).loglikelihood)
        for sv in (None, theta)
    )
    got.update(n0=n0, short=best - ours)
    return got


def lo_model(form: str, r: np.ndarray):  # noqa: ANN201
    vol, dist, _ = LO[form]
    kw = {"garch": dict(vol="GARCH", p=1, o=0, q=1), "gjr": dict(vol="GARCH", p=1, o=1, q=1)}.get(
        vol, dict(vol="EGARCH", p=1, o=1, q=1)
    )
    return arch_model(r, mean="Zero", dist=dist, rescale=False, **kw)


#: Every truth a start is held on: (the truth set, the form).
STARTS = [("hi", f) for f in sorted(FORMS)] + [("lo", f) for f in sorted(LO)]


@pytest.mark.parametrize(("truth", "form"), STARTS)
def test_the_default_start_is_arch_fit_of_its_rows(truth, form):
    """Seed 0 of each truth at the default start of 2,000 rows: its first fit
    does not wait on the evidence, and the fit it reports from (the first,
    or a refit of twice or four times the rows where a fit sat on a bound)
    is short of `arch` by at most 1e-6 nats."""
    got = start_of(truth, form, series_of(truth, form, 8001, 0), 2000)
    assert got["first"] != "not_identified", (truth, form)
    assert got["short"] is not None and got["short"] <= 1e-6, (truth, form, got)


#: Each truth at each start; FIGARCH's start is 2,000 rows at least.
START_CELLS = [
    (truth, form, start)
    for truth, form in STARTS
    for start in (500, 1000, 2000)
    if start == 2000 or FORMS.get(form, ("",))[0] != "figarch"
]


@pytest.mark.extended(reason="50 seeds a truth and a start, a bank and two arch fits each")
@pytest.mark.parametrize(("truth", "form", "start"), START_CELLS)
def test_no_start_is_short_of_arch(truth, form, start):
    gots = [start_of(truth, form, series_of(truth, form, 10_000, s), start) for s in SEEDS]
    shorts = np.array([g["short"] for g in gots if g["short"] is not None])
    assert len(shorts) == len(gots), "every realistic series starts within 10,000 rows"
    if start >= 1000:
        assert shorts.max() <= 1e-6, (truth, form, start, shorts.max())
    else:
        assert (shorts > 0.01).sum() <= 2, (truth, form, np.sort(shorts)[-3:])
    if start == 2000:
        waits = sum(g["first"] == "not_identified" for g in gots)
        assert waits == 0, (truth, form, waits)


def arch_lr(vol: str, dist: str, r: np.ndarray) -> tuple[float, np.ndarray]:
    """`arch`'s likelihood-ratio statistic: its GARCH-family fit against its
    `ConstantVariance` fit, with the same innovations, on the same rows; and
    the fit's process parameters."""
    kw = {"garch": dict(vol="GARCH", p=1, o=0, q=1), "gjr": dict(vol="GARCH", p=1, o=1, q=1)}.get(
        vol, dict(vol="EGARCH", p=1, o=1, q=1)
    )
    full = arch_model(r, mean="Zero", dist=dist, rescale=False, **kw)
    const = arch_model(r, mean="Zero", vol="Constant", dist=dist, rescale=False)
    res = full.fit(disp="off", options={"maxiter": 1000})
    lc = const.fit(disp="off", options={"maxiter": 1000}).loglikelihood
    k = 3 if vol == "garch" else 4
    return 2.0 * float(res.loglikelihood - lc), np.asarray(res.params)[:k]


def in_our_set(vol: str, theta: np.ndarray) -> bool:
    """Whether `arch`'s fit is in the set the bank fits over: EGARCH's box,
    `alpha >= |gamma|`, which `arch` leaves free (a negative news term, whose
    filter need not contract); the others' persistence within the bank's
    margin, `1 - 1e-4`."""
    if vol == "egarch":
        return bool(theta[1] >= abs(theta[2]) and theta[1] <= 1.0 and theta[3] <= 1.0 - 1e-4)
    persistence = theta[1] + theta[2] if vol == "garch" else theta[1] + theta[2] / 2 + theta[3]
    return bool(persistence <= 1.0 - 1e-4)


def lr_cases(seeds: range) -> list[tuple[str, str, int]]:
    return [(t, f, s) for t, f in STARTS if FORMS.get(f, ("",))[0] != "figarch" for s in seeds] + [
        ("iid", f, s) for f in sorted(LO) for s in seeds
    ]


def lr_agreement(cases: list[tuple[str, str, int]]) -> tuple[int, int, list]:
    """Of the 500-row starts, how many `arch`'s test decides clearly (its
    statistic 0.5 or more from the cut, beyond its optimizer's reach, and
    its fit in the bank's set), and those where the bank's wait and `arch`'s
    test differ."""
    clear, differ = 0, []
    for truth, form, seed in cases:
        if truth == "iid":
            vol, dist, _ = LO[form]
            rng = np.random.default_rng([5490, sorted(LO).index(form), seed])
            r = innovations(dist, 600, rng)
        else:
            vol, dist = vol_dist(truth, form)
            r = series_of(truth, form, 600, seed)
        lr, theta = arch_lr(vol, dist, r[:500])
        if abs(lr - LR_CUT[vol]) < 0.5 or not in_our_set(vol, theta):
            continue
        clear += 1
        got = start_of(truth, form, r, 500)
        if (got["first"] == "not_identified") != (lr < LR_CUT[vol]):
            differ.append((truth, form, seed, lr, got["first"]))
    return clear, len(cases), differ


def test_the_wait_is_arch_likelihood_ratio_test():
    """Two seeds of every truth and of returns with no conditional variance,
    at a start of 500 rows, where both sides of the cut are common: the
    bank's first fit waits (`not_identified`) exactly where `arch`'s
    statistic is under the cut."""
    clear, n, differ = lr_agreement(lr_cases(range(2)))
    assert clear >= n // 2 and not differ, (clear, n, differ)


@pytest.mark.extended(reason="20 seeds of 19 truths at 500 rows, two arch fits each")
def test_the_wait_is_arch_likelihood_ratio_test_on_twenty_seeds():
    clear, n, differ = lr_agreement(lr_cases(range(20)))
    assert clear >= n * 3 // 4 and not differ, (clear, n, differ)
