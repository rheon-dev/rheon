"""Task 523: `bocpd`: hazards that depend on run length, and a known-variance emission.

Beside `bocpd.rs`'s recursion (Adams & MacKay Algorithm 1, per-row hazard, `r` the rows in the run before the row):

    growth       P(r+1, x_1:t) = P(r, x_1:t-1) pi_t(r) (1 - H(r))
    changepoint  P(0,   x_1:t) = sum_r P(r, x_1:t-1) pi_t(r) H(r)

Today `H(r) = 1/hazard` for every `r`. New, per C1's verdicts:
  * the negative-binomial hazard (CONFIRMED in `bayesian-changepoint`): segment lengths `P(L = t) = C(t-1, k-1) p^k (1-p)^(t-k)`,
    `H(r) = P(L = r+1) / P(L >= r+1)`: 0 for `r + 1 < k`, `p` at `k = 1`;
  * the known-variance emission (CONFIRMED, `NormalKnownVariance`): the mean `~ N(mu_0, tau^2)`, observation variance `s2` known, a run keeps
    `(n, S = sum x)`; `v = 1/(1/tau^2 + n/s2)`, `m = v (mu_0/tau^2 + S/s2)`, predictive `N(m, v + s2)`.
Not built, MISSING an oracle (C1): the logistic hazard (Turner et al.), and the known-mean (variance-change) emission.

A zero-weight row is a no-op for the recursion: no run grows, no hazard is applied (a per-row hazard counts learned rows).
"""

from __future__ import annotations

import itertools
import math
import warnings
from functools import partial

import numpy as np
from scipy import stats

from .common import table


def nb_hazard(k, p, nmax):
    """H(r), r = 0..nmax-1, from the pmf by scipy (float64): P(L = r+1)/P(L >= r+1)."""
    r = np.arange(nmax)
    pmf = stats.nbinom.pmf(r + 1 - k, k, p) * ((r + 1) >= k)
    sf = stats.nbinom.sf(r - k, k, p)  # P(L >= r+1) = P(failures >= r+1-k) = sf(r-k)
    sf = np.where(r + 1 <= k, 1.0, sf)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(sf > 0, pmf / sf, 1.0)
    return h


def const_hazard(lam, nmax):
    return np.full(nmax, 1.0 / lam)


class KnownVarBocpd:
    """Run-length posterior with a hazard vector `H(r)` and the known-variance normal emission."""

    def __init__(self, s2, mu0, tau2, H):
        self.s2, self.mu0, self.tau2, self.H = s2, mu0, tau2, H
        self.n = np.array([0.0])
        self.S = np.array([0.0])
        self.post = np.array([1.0])
        self.R = [self.post.copy()]

    def step(self, x, w=1.0):
        if w == 0.0:
            return None
        v = 1.0 / (1.0 / self.tau2 + self.n / self.s2)
        m = v * (self.mu0 / self.tau2 + self.S / self.s2)
        pv = v + self.s2
        pred = np.exp(-0.5 * (math.log(2 * math.pi) + np.log(pv) + (x - m) ** 2 / pv))
        r = np.arange(len(self.post))
        H = self.H[r]
        wgt = self.post * pred
        new = np.empty(len(self.post) + 1)
        new[1:] = wgt * (1 - H)
        new[0] = float(np.sum(wgt * H))
        new /= new.sum()
        self.post = new
        self.n = np.concatenate([[0.0], self.n + 1.0])
        self.S = np.concatenate([[0.0], self.S + x])
        self.R.append(new.copy())
        return new

    def state_bytes(self, runs, d=1):
        return 8 * runs * (3 + 2 * d)


def enumerate_posterior(x, s2, mu0, tau2, H):
    """The run-length posterior after len(x) rows by summing every path `r_t in {0, r_{t-1}+1}` (exact)."""
    T = len(x)
    out = np.zeros(T + 1)
    for flags in itertools.product((0, 1), repeat=T):  # 1 = changepoint after row t
        # run rows
        n, S = 0, 0.0
        prob = 1.0
        like = 1.0
        r_prev = 0
        for t in range(T):
            v = 1.0 / (1.0 / tau2 + n / s2)
            m = v * (mu0 / tau2 + S / s2)
            pv = v + s2
            like *= math.exp(-0.5 * (math.log(2 * math.pi) + math.log(pv) + (x[t] - m) ** 2 / pv))
            n, S = n + 1, S + x[t]
            h = H[r_prev]
            if flags[t]:
                prob *= h
                n, S = 0, 0.0
                r_prev = 0
            else:
                prob *= 1 - h
                r_prev += 1
        out[r_prev] += prob * like
    return out / out.sum()


def report(seeds: int = 20) -> None:
    warnings.filterwarnings("ignore")
    import torch
    from bayesian_changepoint_detection import StudentT, constant_hazard, negative_binomial_hazard, online_changepoint_detection
    from bayesian_changepoint_detection.online_likelihoods import NormalKnownVariance

    # ---- the hazard: closed form vs scipy and vs the library's table
    rows = []
    for k, p in ((1, 0.02), (3, 0.05), (5, 0.1), (10, 0.02)):
        n = 300
        h = nb_hazard(k, p, n)
        lib = negative_binomial_hazard(k, p, n).double().numpy()
        # independent: direct pmf ratio with exact sums in float64
        L = np.arange(1, 4000)
        pmf = stats.nbinom.pmf(L - k, k, p) * (L >= k)
        surv = pmf[::-1].cumsum()[::-1]
        h2 = pmf[:n] / surv[:n]
        rows.append([f"k = {k}, p = {p}", float(np.max(np.abs(h - h2))), float(np.max(np.abs(h - lib))), float(h[: k - 1].max()) if k > 1 else float(h[0] - p)])
    table("523a negative-binomial hazard `H(r)` (scipy `nbinom` pmf and survival) vs the pmf ratio from a long cumulative sum (abs), vs `negative_binomial_hazard(k, p, n)` (float32 table, abs); last column: `max H(r)` for `r + 1 < k` (0), or `H(0) - p` at k = 1",
          ["k, p", "vs cumulative sum", "vs library", "H below k / H - p"], rows)

    # ---- exact enumeration, small T
    rows = []
    for label, Hfun in (("constant, lam = 6", lambda n: const_hazard(6, n)), ("NB k = 2, p = 0.3", lambda n: nb_hazard(2, 0.3, n)), ("NB k = 3, p = 0.2", lambda n: nb_hazard(3, 0.2, n))):
        g = 0.0
        g_all = 0.0
        for sd in range(max(3, seeds // 5)):
            rng = np.random.default_rng(sd)
            x = np.concatenate([rng.normal(0, 1, 5), rng.normal(2.5, 1, 5)])
            H = Hfun(20)
            m = KnownVarBocpd(1.0, 0.0, 4.0, H)
            for t in range(len(x)):
                m.step(x[t])
            ref = enumerate_posterior(x, 1.0, 0.0, 4.0, H)
            g = max(g, float(np.max(np.abs(m.post - ref))))
        rows.append([label, g])
    table("523b the recursion vs summing all `2^T` run-length paths from the definition (T = 10, known variance; absolute gap in the run-length posterior)", ["hazard", "largest gap"], rows)

    # ---- the library's run-length posterior (float32)
    rows = []
    for label, Hf, lib_h in (
        ("constant, lam = 100", lambda n: const_hazard(100, n), partial(constant_hazard, 100)),
        ("NB k = 3, p = 0.02", lambda n: nb_hazard(3, 0.02, n), partial(negative_binomial_hazard, 3, 0.02)),
        ("NB k = 10, p = 0.05", lambda n: nb_hazard(10, 0.05, n), partial(negative_binomial_hazard, 10, 0.05)),
    ):
        g = 0.0
        gmap = 0
        for sd in range(max(4, seeds // 4)):
            rng = np.random.default_rng(sd)
            x = np.concatenate([rng.normal(0, 1, 60), rng.normal(3, 1, 60), rng.normal(0, 1, 40)])
            T = len(x)
            m = KnownVarBocpd(1.0, 0.0, 4.0, Hf(T + 5))
            for t in range(T):
                m.step(x[t])
            lik = NormalKnownVariance(variance=1.0, mu=0.0, prior_variance=4.0, device="cpu")
            R, mp = online_changepoint_detection(torch.tensor(x, dtype=torch.float64), lib_h, lik, device="cpu")
            R = R.double().numpy()
            for t in range(T + 1):
                g = max(g, float(np.max(np.abs(m.R[t] - R[: t + 1, t]))))
            gmap += int(sum(int(np.argmax(m.R[t])) != int(mp[t]) for t in range(T + 1)))
        rows.append([label, g, gmap])
    table("523c run-length posterior `R[r, t]` vs `online_changepoint_detection(data, hazard, NormalKnownVariance(variance=1, mu=0, prior_variance=4))` (library state is float32, so the gap is float32's; 160 rows with two level shifts; MAP run lengths that differ out of all columns)",
          ["hazard", "largest gap in R (abs)", "MAP run-length mismatches"], rows)

    # ---- behaviour: NB hazard vs constant on detection delay (the point of the feature)
    rows = []
    for label, Hf in (("constant, lam = 100", lambda n: const_hazard(100, n)), ("NB k = 5, p = 0.05 (mean 100)", lambda n: nb_hazard(5, 0.05, n))):
        # segments of length ~ NB(5, .05): changepoints rarely early
        dl = []
        fa = 0
        for sd in range(60):
            rng = np.random.default_rng(1000 + sd)
            x = np.concatenate([rng.normal(0, 1, 100), rng.normal(2.0, 1, 100)])
            m = KnownVarBocpd(1.0, 0.0, 4.0, Hf(210))
            hit = None
            early = 0
            for t in range(len(x)):
                post = m.step(x[t])
                pc = post[0] + post[1]  # a run at most one row old
                if t < 100 and pc > 0.5 and t > 10:
                    early += 1
                if t >= 100 and hit is None and post[: t - 100 + 3].sum() > 0.5 and np.argmax(post) <= t - 100 + 1:
                    hit = t - 100
            fa += early > 0
            if hit is not None:
                dl.append(hit)
        rows.append([label, np.median(dl) if dl else "", len(dl), fa])
    table("523d what the hazard buys (60 streams: 100 rows N(0,1), then a +2 shift for 100): median detection delay in rows (MAP run length back to the change), streams detected, streams with a spurious early alarm (`P(r <= 1) > 0.5` before the shift)",
          ["hazard", "median delay", "detected / 60", "early alarms / 60"], rows)

    # ---- zero weights
    g = 0.0
    for sd in range(5):
        rng = np.random.default_rng(sd)
        x = rng.normal(0, 1, 80)
        w = (rng.random(80) > 0.2).astype(float)
        H = nb_hazard(3, 0.05, 100)
        a = KnownVarBocpd(1.0, 0.0, 4.0, H)
        b = KnownVarBocpd(1.0, 0.0, 4.0, H)
        for t in range(80):
            a.step(x[t], w[t])
            if w[t] > 0:
                b.step(x[t])
        g = max(g, float(np.max(np.abs(a.post - b.post))))
    print(f"\n523e zero-weight rows (20%) vs the stream without them: run-length posterior gap {g:.1e} (bit-equal: a per-row hazard counts learned rows)")

    # ---- state
    rows = []
    for d in (1, 5, 20):
        rows.append([f"d = {d}", 8 * (3 + 2 * d), 8 * (3 + 3 * d), 262144 // (8 * (3 + 2 * d)), 262144 // (8 * (3 + 3 * d))])
    table("523f state per held run (bytes, from `bocpd.rs`'s `Run`: log-mass, `len`, `n`, then the mean and its low part `2 d` for known variance; `diag` adds the scatter `m2`, `d` more) and the runs the 256 KB per-group cap holds (the default `max_run` is 10,000)",
          ["features", "known-variance run", "diag run", "runs under the cap: known-variance", "runs under the cap: diag"], rows)
    print("  the NB hazard table `H(r)`, `r < max_run`: 8 max_run bytes (80 KB at 10,000), derived from (k, p), shared by every group of the spec, not group state.")
