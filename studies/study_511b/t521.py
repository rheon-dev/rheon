"""Task 521: forecast comparison.

Beside what `rp.eval.diebold_mariano` computes now (squared-loss differential `d = loss_b - loss_a`, positive
when `a` is closer; Newey-West variance with Bartlett weights; a normal reference; two-sided; `half_life=` gives
EW sums). Added, as options, none changing the defaults:

DM with the Harvey-Leybourne-Newbold factor, a t reference, any loss, a one-sided p. With `m` the (EW) mean of `d`,
`W = sum w`, `n` = Kish's `W^2 / sum w^2` and the sum over rows that have a lag-`l` partner, weighted by the later row:

    S = sum w (d - m)^2 + 2 sum_{l=1..L} (1 - l/(L+1)) sum_{i>=l} w_i (d_i - m)(d_{i-l} - m)
    dm = m sqrt(n) / sqrt(S / W)         HLN: dm * sqrt((n + 1 - 2h + h(h-1)/n) / n), h = L + 1
    p: normal (today), or t with n - 1 df; one-sided for "a is closer": sf(dm).

The centred lag products expand around the one global `m`, so each lag keeps four EW sums over the rows with a partner
(`A_l = sum w d_i`, `B_l = sum w d_{i-l}`, `N_l = sum w`, `P_l = sum w d_i d_{i-l}`):
    sum w (d_i - m)(d_{i-l} - m) = P_l - m A_l - m B_l + m^2 N_l.

An anytime-valid test of the mean loss difference: bounded differences `r = (d/B + 1)/2 in [0, 1]`, the
conjugate-mixture empirical-Bernstein confidence sequence of Howard et al. (2021) as MAPIE spells it; reject "a is not
better" when the running lower bound on `E[r]` passes 1/2. No decay, a zero weight skips the row, as `seqtest`.

Giacomini-White's conditional test: `Z_t = h_t d_t` for instruments `h_t = (1, d_{t-1}, ...)`;
    stat = n Zbar' Omega^-1 Zbar ~ chi2(q),  Omega = Gamma_0 + sum_l (1 - l/(L+1)) (Gamma_l + Gamma_l'),  Gamma_l = sum w Z_i Z_{i-l}' / W.
"""

from __future__ import annotations

import math
import warnings
from collections import deque

import numpy as np
from scipy import stats

from .common import dt_stream, lam_of, table


class DM:
    def __init__(self, L):
        self.L = L
        self.W = self.Q = self.sd = self.sdd = 0.0  # sum w, sum w^2, sum w d, sum w d^2
        self.ring = deque(maxlen=L)  # last L scored d
        self.A = np.zeros(L + 1)
        self.B = np.zeros(L + 1)
        self.N = np.zeros(L + 1)
        self.P = np.zeros(L + 1)

    def step(self, d, lam, w):
        self.W *= lam
        self.Q *= lam * lam
        self.sd *= lam
        self.sdd *= lam
        self.A *= lam; self.B *= lam; self.N *= lam; self.P *= lam
        if w == 0.0:
            return
        self.W += w; self.Q += w * w; self.sd += w * d; self.sdd += w * d * d
        past = list(self.ring)[::-1]  # d_{i-1}, d_{i-2}, ...
        for l in range(1, min(self.L, len(past)) + 1):
            dl = past[l - 1]
            self.A[l] += w * d; self.B[l] += w * dl; self.N[l] += w; self.P[l] += w * d * dl
        self.ring.append(d)

    def stat(self, hln=False, ref="normal", one_sided=False):
        W = self.W
        n = W * W / self.Q
        m = self.sd / W
        S = self.sdd - 2 * m * self.sd + m * m * W
        for l in range(1, self.L + 1):
            S += 2 * (1 - l / (self.L + 1)) * (self.P[l] - m * self.A[l] - m * self.B[l] + m * m * self.N[l])
        dm = m * math.sqrt(n) / math.sqrt(S / W)
        h = self.L + 1
        if hln:
            dm *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        sf = stats.t.sf(dm, n - 1) if ref == "t" else stats.norm.sf(dm)
        p = sf if one_sided else 2 * min(sf, 1 - sf)
        return dm, p, n

    def state_bytes(self):
        return 8 * (4 + 4 * (self.L + 1) + self.L) - 8 * 4  # lag-0 A,B,N,P are not kept: 4 + 4L + L scalars


def losses(kind, y, f):
    e = y - f
    if kind == "squared":
        return e * e
    if kind == "absolute":
        return abs(e)
    if kind == "linex":  # asymmetric: LINEX with a = 0.5
        a = 0.5
        return math.exp(a * e) - a * e - 1
    if kind == "pinball":
        t = 0.8
        return max(t * e, (t - 1) * e)
    raise ValueError(kind)


class CMEB:
    """Conjugate-mixture empirical-Bernstein CS (MAPIE's), recursion: S_t, V_t, t, running lower bound."""

    def __init__(self, v_opt=1.0, alpha=0.05, stride=1):
        self.stride = stride
        from mapie.exchangeability_testing.bounds import GammaExponentialMixtureBound

        self.bnd = GammaExponentialMixtureBound(v_opt=v_opt, c=1, alpha_opt=alpha / 2)
        self.alpha = alpha
        self.t = 0
        self.S = 0.0
        self.V = 0.0
        self.mu_prev = 0.5
        self.lower = 0.0
        self.upper = 1.0

    def step(self, r, w=1.0):
        if w == 0.0:
            return
        self.t += 1
        self.S += r
        self.V += (r - self.mu_prev) ** 2
        mu = self.S / self.t
        self.mu_prev = mu
        if self.t % self.stride:
            return  # the boundary is read every `stride` rows; the sums are folded on every row
        bd = self.bnd(v=self.V, alpha=self.alpha / 2) / self.t
        self.lower = max(self.lower, max(mu - bd, 0.0))
        self.upper = min(self.upper, min(mu + bd, 1.0))

    def reject(self):  # mean(r) > 1/2: a is better
        return self.lower > 0.5


class Betting:
    """The longhand: E_t = prod (1 + lam_t d_t), lam_t predictable from the rows before (aGRAPA-style), d in [-1, 1]."""

    def __init__(self):
        self.t = 0
        self.s = 0.0
        self.ss = 0.0
        self.logE = 0.0

    def step(self, d, w=1.0):
        if w == 0.0:
            return
        mu = self.s / self.t if self.t else 0.0
        var = (self.ss / self.t - mu * mu) if self.t > 1 else 0.25
        lam = min(max(mu / (var + mu * mu + 1e-12), 0.0), 0.5)
        self.logE += math.log1p(lam * d)
        self.t += 1
        self.s += d
        self.ss += d * d

    def reject(self, alpha=0.05):
        return self.logE >= math.log(1 / alpha)


class GW:
    def __init__(self, q_extra, L):
        self.q, self.L = 1 + q_extra, L  # instruments: const + q_extra lagged d
        self.ring_d = deque(maxlen=q_extra)
        self.zring = deque(maxlen=L)
        self.W = self.Q = 0.0
        self.zs = np.zeros(self.q)
        self.Gam = np.zeros((L + 1, self.q, self.q))
        self.q_extra = q_extra

    def step(self, d, lam, w):
        self.W *= lam; self.Q *= lam * lam; self.zs *= lam; self.Gam *= lam
        if w == 0.0:
            return
        if len(self.ring_d) < self.q_extra:
            self.ring_d.append(d)
            return
        h = np.concatenate([[1.0], list(self.ring_d)[::-1]]) if self.q_extra else np.array([1.0])
        z = h * d
        self.W += w; self.Q += w * w; self.zs += w * z
        self.Gam[0] += w * np.outer(z, z)
        past = list(self.zring)[::-1]
        for l in range(1, min(self.L, len(past)) + 1):
            self.Gam[l] += w * np.outer(z, past[l - 1])
        self.zring.append(z)
        if self.q_extra:
            self.ring_d.append(d)

    def stat(self):
        n = self.W * self.W / self.Q
        zbar = self.zs / self.W
        Om = self.Gam[0] / self.W
        for l in range(1, self.L + 1):
            Om = Om + (1 - l / (self.L + 1)) * (self.Gam[l] + self.Gam[l].T) / self.W
        return n * float(zbar @ np.linalg.solve(Om, zbar)), n

    def state_bytes(self):
        q, L = self.q, self.L
        return 8 * ((L + 1) * q * q + q * L + q + self.q_extra + 2)


def report(seeds: int = 20) -> None:
    warnings.filterwarnings("ignore")
    from dieboldmariano import dm_test

    # -------- (a) DM vs dieboldmariano, lam=1 unit w
    rows = []
    for kind in ("squared", "absolute", "linex", "pinball"):
        for L in (0, 2, 5):
            gs = gp = gp1 = gph = 0.0
            for sd in range(seeds):
                rng = np.random.default_rng(sd)
                n = 200
                y = rng.standard_normal(n)
                fa = y + 0.9 * rng.standard_normal(n)
                fb = y + 1.0 * rng.standard_normal(n) + 0.05
                d = np.array([losses(kind, y[t], fb[t]) - losses(kind, y[t], fa[t]) for t in range(n)])  # positive: a closer
                m = DM(L)
                for t in range(n):
                    m.step(d[t], 1.0, 1.0)
                h = L + 1
                lf = (lambda u, v, kind=kind: losses(kind, u, v))
                for hln in (False, True):
                    st, p, _ = m.stat(hln=hln, ref="t")
                    rst, rp = dm_test(y, fa, fb, loss=lf, h=h, harvey_correction=hln, variance_estimator="bartlett")
                    gs = max(gs, abs(st + rst) / max(1.0, abs(rst)))  # the oracle's d is loss_a - loss_b
                    gp = max(gp, abs(p - rp))
                    st1, p1, _ = m.stat(hln=hln, ref="t", one_sided=True)
                    _, rp1 = dm_test(y, fa, fb, loss=lf, h=h, harvey_correction=hln, one_sided=True, variance_estimator="bartlett")
                    gp1 = max(gp1, abs(p1 - rp1))
            rows.append([f"{kind}, lags {L} (h = {h})", gs, gp, gp1])
    table("521a DM statistic (HLN on and off), two-sided p with the t reference and one-sided p ('a closer') vs `dm_test(..., variance_estimator=\"bartlett\", h=lags+1)` (the oracle's sign is `loss_a - loss_b`; 200 rows)",
          ["loss, lags", "statistic (rel)", "two-sided p (abs)", "one-sided p (abs)"], rows)

    # -------- (b) decay, weights, zero weights vs two-pass definition and folded
    g = {"stat": 0.0, "fold": 0.0}
    for sd in range(seeds):
        n = 220
        rng = np.random.default_rng(sd)
        d = rng.standard_normal(n) * 1.0 + 0.1
        dt, w = dt_stream(sd, n)
        L = 3
        m = DM(L)
        for t in range(n):
            m.step(d[t], 1.0 if t == 0 else lam_of(dt[t], 40.0), w[t])
        st = m.stat(hln=True, ref="t")[0]
        lam_ = [1.0 if t == 0 else lam_of(dt[t], 40.0) for t in range(n)]
        om = np.zeros(n); acc = 1.0
        for t in range(n - 1, -1, -1):
            om[t] = w[t] * acc; acc *= lam_[t]
        idx = [t for t in range(n) if w[t] > 0]
        D = d[idx]; o = om[idx]
        W = o.sum(); mean = o @ D / W
        S = o @ (D - mean) ** 2
        for l in range(1, L + 1):
            S += 2 * (1 - l / (L + 1)) * np.sum(o[l:] * (D[l:] - mean) * (D[:-l] - mean))
        nk = W * W / (o**2).sum()
        ref = mean * math.sqrt(nk) / math.sqrt(S / W) * math.sqrt((nk + 1 - 2 * (L + 1) + (L + 1) * L / nk) / nk)
        g["stat"] = max(g["stat"], abs(st - ref) / abs(ref))
        m2 = DM(L)
        carry = 0.0; first = True
        for t in range(n):
            carry += 0.0 if t == 0 else dt[t]
            if w[t] > 0:
                m2.step(d[t], 1.0 if first else lam_of(carry, 40.0), w[t]); first = False; carry = 0.0
        g["fold"] = max(g["fold"], abs(st - m2.stat(hln=True, ref="t")[0]) / abs(st))
    table("521b DM under decay (half-life 40), weights, 10% zero-weight rows, a zero-weight first row (statistic with HLN, lags 3)",
          ["comparison", "largest relative gap"], [["vs the two-pass weighted definition", g["stat"]], ["vs the same stream, zero-weight rows removed and gaps folded", g["fold"]]])

    # -------- (c) size under H0 at half-life
    rows = []
    nrep = 400
    for hl in (10.0, 30.0, 100.0):
        for L in (0, 4):
            rej = {"normal": 0, "normal+HLN": 0, "t": 0, "t+HLN": 0}
            for sd in range(nrep):
                rng = np.random.default_rng(9000 + sd)
                n = 400
                d = rng.standard_normal(n)
                m = DM(L)
                lam = 0.5 ** (1.0 / hl)
                for t in range(n):
                    m.step(d[t], lam, 1.0)
                for key, (hln, ref) in {"normal": (False, "normal"), "normal+HLN": (True, "normal"), "t": (False, "t"), "t+HLN": (True, "t")}.items():
                    rej[key] += m.stat(hln=hln, ref=ref)[1] < 0.05
            rows.append([f"half-life {hl:g}, lags {L}", *(rej[k] / nrep for k in ("normal", "normal+HLN", "t", "t+HLN"))])
    table("521c share of 400 null streams (iid differentials, 400 rows) the final-row two-sided test passed at 5% (binomial sd 0.011); `n` = Kish",
          ["case", "normal (today)", "normal + HLN", "t, n-1", "t + HLN"], rows)

    # -------- (d) CM-EB vs MAPIE
    from mapie.exchangeability_testing.bounds import conjugate_mixture_empirical_bernstein_bound as cm

    gl = gu = 0.0
    for sd in range(max(5, seeds // 4)):
        rng = np.random.default_rng(sd)
        r = np.clip(0.5 + 0.08 + 0.2 * rng.standard_normal(150), 0, 1)
        lo = cm(r, v_opt=1.0, alpha=0.05, bound_side="lower")
        up = cm(r, v_opt=1.0, alpha=0.05, bound_side="upper")
        c = CMEB(1.0, 0.05)
        for t in range(150):
            c.step(r[t])
            gl = max(gl, abs(c.lower - lo[t])); gu = max(gu, abs(c.upper - up[t]))
    print(f"\n521d CM-EB recursion vs `conjugate_mixture_empirical_bernstein_bound` (lower / upper, running intersection, 150 rows): {gl:.1e} / {gu:.1e}")
    # type I and power under bounded differences: d in [-1,1], r = (d+1)/2
    rows = []
    for label, shift in (("H0: E[d] = 0 (the null's edge)", 0.0), ("H0: E[d] = -0.05", -0.05), ("H1: E[d] = +0.10", 0.10), ("H1: E[d] = +0.20", 0.20)):
        rej_c = rej_b = 0
        t_c = []
        t_b = []
        nrep = 100
        T = 1500
        for sd in range(nrep):
            rng = np.random.default_rng(777 + sd)
            d = np.clip(shift + 0.35 * rng.standard_normal(T), -1, 1)
            if shift == 0.0:
                d = d - d.mean() * 0  # clipping is symmetric: mean stays 0
            c = CMEB(100.0, 0.05, stride=10)
            b = Betting()
            hit_c = hit_b = None
            for t in range(T):
                c.step((d[t] + 1) / 2)
                b.step(d[t])
                if hit_c is None and t % 10 == 9 and c.reject():
                    hit_c = t
                if hit_b is None and b.reject():
                    hit_b = t
            rej_c += hit_c is not None
            rej_b += hit_b is not None
            if hit_b is not None:
                t_b.append(hit_b)
            if hit_c is not None:
                t_c.append(hit_c)
        rows.append([label, rej_c / nrep, rej_b / nrep, int(np.median(t_c)) if t_c else "", int(np.median(t_b)) if t_b else ""])
    table("521e anytime-valid test of the mean loss difference: share of 100 streams of 1500 rows that EVER rejected (CM-EB `v_opt = 100` at 5%, read every 10 rows; the betting e-process at 5%, read every row); median stopping row of CM-EB",
          ["stream", "CM-EB", "betting e-process", "CM-EB median stop", "betting median stop"], rows)

    # -------- (e) GW vs statsmodels
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import S_hac_simple, weights_bartlett

    rows = []
    for dgp in ("iid, mean 0 (H0 true)", "AR(.3) (H0 false: d_t predicted by d_{t-1})"):
        for q_extra, L in ((0, 0), (1, 3), (2, 5)):
            g_exact = g_wald = 0.0
            for sd in range(seeds):
                rng = np.random.default_rng(sd)
                n = 300
                d = rng.standard_normal(n)
                if dgp.startswith("AR"):
                    for t in range(1, n):
                        d[t] += 0.3 * d[t - 1]
                gwm = GW(q_extra, L)
                for t in range(n):
                    gwm.step(d[t], 1.0, 1.0)
                st, nn = gwm.stat()
                y = d[q_extra:]
                H = np.column_stack([np.ones(len(y))] + [d[q_extra - j - 1 : n - j - 1] for j in range(q_extra)])
                Z = H * y[:, None]
                S = S_hac_simple(Z, nlags=L, weights_func=weights_bartlett)  # sum of w_l Gamma_l, uncentered
                zs = Z.sum(0)
                ref_exact = float(zs @ np.linalg.solve(S, zs))
                g_exact = max(g_exact, abs(st - ref_exact) / ref_exact)
                res = sm.OLS(y, H).fit(cov_type="HAC", cov_kwds={"maxlags": L, "kernel": "bartlett", "use_correction": False})
                wt = float(res.wald_test(np.eye(H.shape[1]), use_f=False, scalar=True).statistic)
                g_wald = max(g_wald, abs(st - wt) / wt)
            rows.append([f"{dgp}, q = {1 + q_extra}, lags {L}", g_exact, g_wald])
    table("521f Giacomini-White's statistic: vs the definition by statsmodels `S_hac_simple(Z, nlags=L)` (`Z_t = h_t d_t`, Bartlett, uncentered) and vs `OLS(d, h).fit(cov_type=\"HAC\", maxlags=L, kernel=\"bartlett\").wald_test(I, use_f=False)` (relative gaps, 300 rows)",
          ["case", "vs definition (S_hac_simple)", "vs the HAC Wald"], rows)
    # size of GW under decay
    rows = []
    for q_extra, L in ((0, 0), (1, 3)):
        rej = 0
        nrep = 500
        for sd in range(nrep):
            rng = np.random.default_rng(31000 + sd)
            n = 500
            d = rng.standard_normal(n)
            gwm = GW(q_extra, L)
            lam = 0.5 ** (1 / 100.0)
            for t in range(n):
                gwm.step(d[t], lam, 1.0)
            st, nn = gwm.stat()
            rej += st > stats.chi2.ppf(0.95, 1 + q_extra)
        rows.append([f"q = {1 + q_extra}, lags {L}, half-life 100 rows, iid", rej / nrep])
    table("521g GW size at the 5% chi-square value (500 null streams; binomial sd 0.010)", ["case", "share passing"], rows)

    print("\n521 state added, per slot (bytes):")
    srows = [[f"DM, lags {L}", DM(L).state_bytes()] for L in (0, 4, 10, 30)]
    srows.append(["CM-EB / betting e-process", "CM-EB: S, V, t, mu_prev, running lower, running upper = 48; betting: 3 sums + log E = 32"])
    for q, L in ((1, 0), (2, 3), (4, 10), (8, 30)):
        srows.append([f"GW, q = {q} instruments, lags {L}", GW(q - 1, L).state_bytes()])
    table("521h state bytes", ["feature", "bytes"], srows)
