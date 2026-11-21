"""Task 515: asymmetric and multi-level conformal intervals, and a per-row p-value.

The existing tracker (one scalar `m` on the multiplier of the EW residual sd
`sigma`, radius `m sigma`, symmetric, one level) steps

    err = 1{|r| > m sigma}      m' = max(0, m + eta (w / w_bar) (err - alpha))

**Asymmetric, several levels.** One pair of *signed* trackers per level
`alpha_l`, each a quantile tracker of the signed scaled residual
`s = r / sigma` at the tail level `alpha_l / 2` (MAPIE's `sym=False` with
`beta = alpha / 2`):

    hi:  err = 1{ s >  m_hi}   m_hi' = m_hi + eta rel (err - alpha_l/2)
    lo:  err = 1{-s >  m_lo}   m_lo' = m_lo + eta rel (err - alpha_l/2)
    interval  [pred - m_lo sigma, pred + m_hi sigma],   rel = w / w_bar

No clamp at 0 (a skewed stream can put one side's quantile on the other side
of the prediction). Both start at `Phi^-1(1 - alpha_l/2)`. The levels are
independent recursions, so nested levels can cross; they are sorted when read
(no state). Telescoping gives, exactly and for every sequence,

    m_{T+1} - m_1 = eta sum_t rel_t (err_t - alpha_l/2)

and `m` stays in `[-B - eta, B + eta]` for scaled scores in `[-B, B]`, which
bounds each side's long-run miss rate within `alpha_l/2` by `2 (B + eta) / (eta T)`.

**Per-row conformal p-value** (crepes' `predict_p_online`, `smoothing=False`):
the share of past scores `a_i = |r_i| / sigma_i` at least as large as the row's,
the row itself counted once,

    p_t = (sum_i omega_i 1{a_i >= a_t} + 1) / (sum_i omega_i + 1),

omega_i the row's weight at the present time (`w_i lam^age`; all 1 at no decay
and unit weights, where it is crepes' count form). A rank needs the scores, so
the state is a ring of the last `W` of them with their clock stamps: 16 W
bytes. A zero-weight row has no p-value and is not stored; the clock still moves.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import norm

from .common import lam_of


class EwVar:
    """EW mean-form variance of the residual, read before the row."""

    def __init__(self):
        self.W = 0.0
        self.m = 0.0
        self.v = 0.0

    def sigma(self):
        return np.sqrt(self.v) if self.W > 0 and self.v > 0 else np.nan

    def update(self, r, lam, w):
        Wo = lam * self.W
        W2 = Wo + w
        self.W = W2
        if w == 0.0 or W2 == 0.0:
            return
        a, b = Wo / W2, w / W2
        d = r - self.m
        self.v = a * self.v + a * b * d * d
        self.m += b * d


class SignedConformal:
    def __init__(self, levels, rate=0.02):
        self.alpha = 1 - np.asarray(levels, float)
        self.rate = rate
        self.start = norm.ppf(1 - self.alpha / 2)
        self.m_hi = self.start.copy()
        self.m_lo = self.start.copy()
        self.cov = np.zeros(len(levels))
        self.cov_w = 0.0
        self.w_mean = 0.0
        self.w_rows = 0.0
        self.started = False
        self.ident = np.zeros((2, len(levels)))  # sum rel (err - alpha/2), to check the telescoping
        self.rows = 0

    def interval(self, pred, sigma):
        """Raw (unsorted) bounds; `crossed` below counts rows where nested levels cross."""
        return pred - self.m_lo * sigma, pred + self.m_hi * sigma

    def update(self, r, sigma, lam, w):
        if not np.isfinite(r) or w <= 0:
            self.cov_w *= lam
            self.w_rows *= lam
            return
        self.w_rows = lam * self.w_rows + 1.0
        self.w_mean += (w - self.w_mean) / self.w_rows
        if not self.started:  # the first row starts the tracker and is not scored
            self.started = True
            self.cov_w *= lam
            return
        if not (np.isfinite(sigma) and sigma > 0):
            self.cov_w *= lam
            return
        rel = w / self.w_mean
        s = r / sigma
        e_hi = (s > self.m_hi).astype(float)
        e_lo = (-s > self.m_lo).astype(float)
        hit = ((s <= self.m_hi) & (-s <= self.m_lo)).astype(float)
        cw = lam * self.cov_w + w
        self.cov = (lam * self.cov_w * self.cov + w * hit) / cw
        self.cov_w = cw
        step = self.rate * rel
        self.m_hi = self.m_hi + step * (e_hi - self.alpha / 2)
        self.m_lo = self.m_lo + step * (e_lo - self.alpha / 2)
        self.ident[0] += rel * (e_hi - self.alpha / 2)
        self.ident[1] += rel * (e_lo - self.alpha / 2)
        self.rows += 1

    @staticmethod
    def state_bytes(levels: int) -> int:
        """Per slot: m_hi, m_lo, realized coverage per level (24 L) + shared cov_w, w_mean, w_rows (24)."""
        return 24 * levels + 24


class PValueRing:
    def __init__(self, size):
        self.size = size
        self.a = deque(maxlen=size)
        self.t = deque(maxlen=size)
        self.w = deque(maxlen=size)

    def p(self, a_t, now, half_life):
        """Read before the row is stored (rule 2)."""
        if not self.a:
            return np.nan
        a = np.asarray(self.a)
        om = np.asarray(self.w) * np.array([lam_of(now - t, half_life) for t in self.t])
        return (np.sum(om[a >= a_t]) + 1.0) / (om.sum() + 1.0)

    def push(self, a_t, now, w):
        if w > 0:
            self.a.append(a_t)
            self.t.append(now)
            self.w.append(w)

    def state_bytes(self):
        return 16 * self.size  # score and clock stamp (the weight is the row's: 8 more if weighted)


# ---------------------------------------------------------------------------
def skewed_stream(seed, n, k=3):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    beta = rng.standard_normal(k)
    noise = (rng.gamma(2.0, 1.0, n) - 2.0) * 0.7
    y = x @ beta + noise
    return x, y, beta, noise


def report(seeds: int = 20) -> None:
    from crepes import ConformalRegressor
    from mapie.conformity_scores import AbsoluteConformityScore
    from mapie.regression import SplitConformalRegressor
    from sklearn.linear_model import LinearRegression

    from .common import table

    levels = [0.8, 0.9, 0.95]
    # (a) the recursion against MAPIE SplitConformalRegressor(sym=False, confidence_level=[...])
    rows, rows_t = [], []
    for rate in (0.05, 0.02, 0.005):
        gap = np.zeros((2, len(levels)))
        tru = np.zeros((4, len(levels)))  # tracker, MAPIE: gap to the true noise quantile, lo and hi
        from scipy.stats import gamma as _g
        a2 = (1 - np.array(levels)) / 2
        t_lo, t_hi = (_g.ppf(a2, 2.0) - 2.0) * 0.7, (_g.ppf(1 - a2, 2.0) - 2.0) * 0.7
        cov = np.zeros(len(levels))
        ident = 0.0
        cross = 0.0
        ns = min(seeds, 8)
        for sd in range(ns):
            n = 40000
            x, y, beta, _ = skewed_stream(sd, n)
            pred = x @ beta  # the model's out-of-sample prediction (a prefit linear model)
            ev = EwVar()
            tr = SignedConformal(levels, rate)
            lo_s, hi_s = [], []
            sig_s = []
            for i in range(n):
                r = y[i] - pred[i]
                sg = ev.sigma()
                if i >= n // 2 and np.isfinite(sg):
                    lo, hi = tr.interval(pred[i], sg)
                    lo_s.append(pred[i] - lo)
                    hi_s.append(hi - pred[i])
                    sig_s.append(sg)
                    cross += float(np.any(np.diff(hi) < 0) or np.any(np.diff(lo) > 0))
                tr.update(r, sg, 1.0, 1.0)
                ev.update(r, 1.0, 1.0)
            # MAPIE: prefit estimator, calibrated on an independent draw of the same distribution
            xc, _, _, nc = skewed_stream(1000 + sd, 40000)
            yc = xc @ beta + nc  # the same model, fresh rows
            est = LinearRegression().fit(x, y)
            est.coef_, est.intercept_ = beta, 0.0
            sc = SplitConformalRegressor(estimator=est, confidence_level=levels,
                                         conformity_score=AbsoluteConformityScore(sym=False), prefit=True)
            sc.conformalize(xc, yc)
            _, iv = sc.predict_interval(np.zeros((1, 3)))
            # an interval about a zero prediction: [lo_off, hi_off] per level
            lo_off, hi_off = iv[0, 0, :], iv[0, 1, :]
            gap[0] = np.maximum(gap[0], np.abs(np.mean(lo_s, 0) + lo_off))
            gap[1] = np.maximum(gap[1], np.abs(np.mean(hi_s, 0) - hi_off))
            tru[0] = np.maximum(tru[0], np.abs(np.mean(lo_s, 0) + t_lo))
            tru[1] = np.maximum(tru[1], np.abs(lo_off - t_lo))
            tru[2] = np.maximum(tru[2], np.abs(np.mean(hi_s, 0) - t_hi))
            tru[3] = np.maximum(tru[3], np.abs(hi_off - t_hi))
            cov = np.maximum(cov, np.abs(tr.cov - np.array(levels)))
            ident = max(ident, float(np.max(np.abs(np.array([tr.m_hi, tr.m_lo]) - tr.start - rate * tr.ident))))
        rows.append([rate, *gap[0], *gap[1], float(cov.max()), ident, cross / (ns * n / 2)])
        rows_t.append([rate, *tru[2], *tru[3], *tru[0], *tru[1]])
    table("515a: late-stream tracked offsets (mean of the second half) vs MAPIE SplitConformalRegressor(AbsoluteConformityScore(sym=False), confidence_level=[.8,.9,.95]) calibrated on 40000 fresh rows; skewed noise (gamma, sd 0.99); absolute gap in units of the noise (sd ~ 1)",
          ["rate", "lo .8", "lo .9", "lo .95", "hi .8", "hi .9", "hi .95", "max |realized cov - level|", "telescoping identity gap", "rows with crossed levels"], rows)

    table("515a2: the same offsets against the true noise quantiles (gamma(2) centred, scale 0.7; largest gap over seeds): the tracker and MAPIE (40000 calibration rows) per side and level. MAPIE's own sampling error is of the same size as the tracker-vs-MAPIE gap above",
          ["rate", "tracker hi .8", ".9", ".95", "MAPIE hi .8", ".9", ".95", "tracker lo .8", ".9", ".95", "MAPIE lo .8", ".9", ".95"], rows_t)

    # (b) telescoping bound: observed miss-rate deviation against 2 (B + eta) / (eta T)
    rows = []
    for rate in (0.05, 0.005):
        for T in (1000, 20000):
            dev = bnd = 0.0
            for sd in range(min(seeds, 5)):
                x, y, beta, _ = skewed_stream(sd, T + 1)
                pred = x @ beta
                ev = EwVar()
                tr = SignedConformal([0.9], rate)
                B = 0.0
                errs = np.zeros(2)
                scored = 0
                for i in range(T + 1):
                    r = y[i] - pred[i]
                    sg = ev.sigma()
                    if tr.started and np.isfinite(sg):
                        B = max(B, abs(r / sg))
                        errs += [r / sg > tr.m_hi[0], -r / sg > tr.m_lo[0]]
                        scored += 1
                    tr.update(r, sg, 1.0, 1.0)
                    ev.update(r, 1.0, 1.0)
                dev = max(dev, float(np.max(np.abs(errs / scored - 0.05))))
                bnd = max(bnd, 2 * (B + rate) / (rate * scored))
            rows.append([rate, T, dev, bnd])
    table("515b: each side's miss rate against alpha/2 = 0.05 (largest deviation over seeds) and the telescoping bound 2 (B + eta) / (eta T), B the largest scaled score seen",
          ["rate", "T", "observed", "bound"], rows)

    # (c) decay, weights and zero-weight rows on the signed trackers
    rows = []
    gz = 0.0
    fin = True
    for sd in range(min(seeds, 6)):
        n = 3000
        x, y, beta, _ = skewed_stream(sd, n)
        pred = x @ beta
        rng = np.random.default_rng(sd)
        dt = rng.uniform(0.5, 1.5, n)
        w = rng.uniform(0.5, 2.0, n)
        zero = np.zeros(n, bool)
        zero[7::6] = True
        zero[0] = True
        a, b = SignedConformal([0.9]), SignedConformal([0.9])
        ea, eb = EwVar(), EwVar()
        carry = 0.0
        for i in range(n):
            r = y[i] - pred[i]
            lam = lam_of(dt[i], 30.0)
            if zero[i]:
                a.update(r, ea.sigma(), lam, 0.0)
                ea.update(r, lam, 0.0)
                carry += dt[i]
                continue
            a.update(r, ea.sigma(), lam, w[i])
            ea.update(r, lam, w[i])
            lam2 = lam_of(dt[i] + carry, 30.0)
            carry = 0.0
            b.update(r, eb.sigma(), lam2, w[i])
            eb.update(r, lam2, w[i])
        fin &= bool(np.isfinite(a.m_hi).all() and np.isfinite(a.m_lo).all())
        gz = max(gz, float(np.max(np.abs(np.r_[a.m_hi - b.m_hi, a.m_lo - b.m_lo]))))
    rows.append(["half_life 30, w in [.5,2], zero-weight rows (first row too) vs their gaps folded forward", gz, fin])
    table("515c: zero-weight rows (largest gap in the multipliers; they differ by the rounding of lam^a lam^b)",
          ["case", "gap", "all finite"], rows)

    # (d) per-row p-value vs crepes predict_p_online (expanding) and predict_p (fixed calibration set)
    rows = []
    for sd in range(min(seeds, 5)):
        n0, n = 60, 400
        x, y, beta, _ = skewed_stream(sd, n0 + n)
        pred = x @ beta
        sig = 0.8 + 0.5 * np.abs(np.sin(np.arange(n0 + n) / 30))
        res = y - pred
        al = np.abs(res) / sig
        cr = ConformalRegressor()
        cr.fit(res[:n0], sigmas=sig[:n0])
        pc = cr.predict_p_online(pred[n0:], y[n0:], sigmas=sig[n0:], smoothing=False, warm_start=True)
        ring = PValueRing(10**6)
        for i in range(n0):
            ring.push(al[i], float(i), 1.0)
        ours = []
        for i in range(n0, n0 + n):
            ours.append(ring.p(al[i], float(i), np.inf))
            ring.push(al[i], float(i), 1.0)
        g_exp = float(np.max(np.abs(np.asarray(ours) - np.asarray(pc))))
        W = 100
        ring = PValueRing(W)
        for i in range(n0 + n - 1):
            ring.push(al[i], float(i), 1.0)
        cw = ConformalRegressor().fit(res[n0 + n - 1 - W : n0 + n - 1], sigmas=sig[n0 + n - 1 - W : n0 + n - 1])
        pw = cw.predict_p(pred[-1:], y[-1:], sigmas=sig[-1:], smoothing=False)
        g_win = abs(ring.p(al[-1], float(n0 + n - 1), np.inf) - float(np.ravel(pw)[0]))
        # decay: explicit weighted share
        hl = 25.0
        ring = PValueRing(10**6)
        w = np.random.default_rng(sd).uniform(0.5, 2.0, n0 + n)
        w[::9] = 0.0
        for i in range(n0 + n - 1):
            ring.push(al[i], float(i), w[i])
        i = n0 + n - 1
        om = np.array([w[j] * 0.5 ** ((i - j) / hl) for j in range(i)])
        want = (om[al[:i] >= al[i]].sum() + 1) / (om.sum() + 1)
        g_dec = abs(ring.p(al[i], float(i), hl) - want)
        rows.append([sd, g_exp, g_win, g_dec])
    table("515d: per-row p-value, no smoothing: expanding vs crepes predict_p_online; a 100-row window vs crepes predict_p on that calibration set; EW-weighted vs the explicit weighted share (absolute gap)",
          ["seed", "expanding", "window 100", "decay + weights + zero-weight rows"], rows)

    rows = [[L, SignedConformal.state_bytes(L), 40 + 0] for L in (1, 3, 5)]
    rows += [[f"p-value ring W={W}", PValueRing(W).state_bytes(), ""] for W in (256, 1024, 8192)]
    table("515e: state per slot: signed trackers 24 L + 24 bytes at L levels (the symmetric tracker today: m, cov and its three shared sums, 40); p-value ring 16 W (+8 W if row weights are not all 1)",
          ["levels / ring", "bytes", "symmetric single level today"], rows)
