"""Task 513: streaming scores for intervals and losses.

Every score that is a mean of a per-row quantity `s_t` (coverage `1{lo<=y<=hi}`,
width, Winkler, pinball, absolute error, log loss) is one EW mean, in the
mean form the models use (`W' = lam W + w`, `a = lam W / W'`, `b = w / W'`,
`m' = a m + b s`): 8 bytes of state per score and one `W` shared. The row's
score is read *before* the row is folded into its own mean: the model's `pred`
and interval for the row were made before the row's target was known (rule 2),
and the score is taken on that out-of-sample output.

    coverage:  s = 1{min(lo,hi) <= y <= max(lo,hi)}   (MAPIE does not un-cross; see report)
    width:     s = |hi - lo|
    Winkler:   s = (u - l) + (2/alpha) ((l - y)+ + (y - u)+),  l = min(lo,hi), u = max(lo,hi)
    pinball:   s = max(a (y - yhat), (a - 1)(y - yhat))
    abs error: s = |y - yhat|
    log loss:  s = -(y ln p' + (1-y) ln(1-p')),  p' = clip(p, eps, 1-eps), eps = float64 eps

Not a mean of a per-row quantity, so not one scalar:

* size-stratified coverage: intervals ranked by width, split into equal-count
  groups (MAPIE's `regression_ssc`): needs the window's own widths. Held for
  a row window: a ring of (width, hit), 16 bytes a row.
* D2 pinball: 1 - sum rho(y - yhat) / sum rho(y - q), q the empirical
  alpha-quantile of the *whole* y_true (scikit-learn: a hindsight null model).
  Exact over a row window from a ring of (y, rho_t) (16 bytes a row) and a sorted read of y
  (plus an order index, 8 bytes a row); over an EW stream the null can only be
  a running quantile, a different quantity (8 bytes: measured in 513d).
"""

from __future__ import annotations

import numpy as np

from .common import lam_of

EPS = np.finfo(np.float64).eps


class EwScores:
    """A vector of EW means over a shared weight."""

    def __init__(self, n: int):
        self.W = 0.0
        self.m = np.zeros(n)

    def update(self, s, lam: float, w: float) -> None:
        Wo = lam * self.W
        W2 = Wo + w
        self.W = W2
        if w == 0.0 or W2 == 0.0 or not np.all(np.isfinite(s)):
            return
        b = w / W2
        self.m = (1 - b) * self.m + b * np.asarray(s, float)

    def state_bytes(self) -> int:
        return 8 * (1 + len(self.m))


def row_scores(y, lo, hi, alpha, pred, quantiles, p_hat=None, y_bin=None):
    l, u = min(lo, hi), max(lo, hi)
    cov = 1.0 if l <= y <= u else 0.0
    wid = u - l
    wink = wid + (2 / alpha) * (max(l - y, 0.0) + max(y - u, 0.0))
    pin = [max(q * (y - pred), (q - 1) * (y - pred)) for q in quantiles]
    return np.array([cov, wid, wink, abs(y - pred), *pin])


def pinball(r, a):
    return np.maximum(a * r, (a - 1) * r)


def sk_null_quantile(y, a):
    """scikit-learn's `_weighted_percentile(average=True)` at unit weights: the
    inverted-CDF quantile, averaged with the next value where alpha n is whole."""
    ys = np.sort(y)
    n = len(ys)
    j = a * n
    if abs(j - round(j)) < 1e-9 and 0 < round(j) < n:
        return 0.5 * (ys[round(j) - 1] + ys[round(j)])
    return ys[min(max(int(np.ceil(j)) - 1, 0), n - 1)]


class Window:
    """Ring of the last `size` rows' (width, hit, y, rho): SSC and D2 over a window."""

    def __init__(self, size: int, alpha_q: float):
        self.size = size
        self.a = alpha_q
        self.w, self.h, self.y, self.rho = [], [], [], []

    def push(self, wid, hit, y, rho):
        for buf, v in ((self.w, wid), (self.h, hit), (self.y, y), (self.rho, rho)):
            buf.append(v)
            if len(buf) > self.size:
                buf.pop(0)

    def ssc(self, bins=3):
        order = np.argsort(self.w, kind="stable")
        return np.array([np.mean(np.asarray(self.h)[idx]) for idx in np.array_split(order, bins)])

    def d2(self):
        y = np.asarray(self.y)
        q = sk_null_quantile(y, self.a)
        den = np.sum(pinball(y - q, self.a))
        return 1 - np.sum(self.rho) / den

    def state_bytes(self):
        return 8 * 4 * self.size + 8 * self.size  # four rings plus an order index


# ---------------------------------------------------------------------------
def make_rows(seed, n, stationary=False):
    rng = np.random.default_rng(seed)
    if stationary:  # AR(1) mean: the stream's distribution does not move
        e = rng.standard_normal(n)
        mu = np.zeros(n)
        for i in range(1, n):
            mu[i] = 0.9 * mu[i - 1] + 0.6 * e[i]
    else:
        mu = np.cumsum(rng.standard_normal(n)) * 0.05
    sd = 0.8 + 0.6 * np.abs(np.sin(np.arange(n) / 40))
    y = mu + sd * rng.standard_normal(n)
    pred = mu + 0.3 * rng.standard_normal(n)
    s = sd * rng.uniform(0.5, 2.0, n)
    lo, hi = pred - 1.64 * s, pred + 1.64 * s
    swap = rng.random(n) < 0.05  # crossed bounds
    lo[swap], hi[swap] = hi[swap], lo[swap]
    p = np.clip(1 / (1 + np.exp(-(0.8 * mu + 0.3 * rng.standard_normal(n)))), 1e-3, 1 - 1e-3)
    p[rng.random(n) < 0.01] = 1.0  # a probability of exactly 1, to meet the clip
    yb = (rng.random(n) < 1 / (1 + np.exp(-0.8 * mu))).astype(float)
    return y, pred, lo, hi, p, yb


def report(seeds: int = 20) -> None:
    from mapie.metrics.regression import (regression_coverage_score, regression_mean_width_score,
                                          regression_mwi_score, regression_ssc)
    from sklearn.metrics import d2_pinball_score, log_loss, mean_absolute_error, mean_pinball_loss

    from .common import stream, table, ulps

    alpha, quants = 0.1, (0.1, 0.5, 0.9)
    names = ["coverage", "width", "Winkler", "MAE"] + [f"pinball {q}" for q in quants]

    def oracle_vec(y, pred, lo, hi, om):
        """The oracle's value on rows weighted by om: sklearn takes sample_weight;
        MAPIE has none, so its per-row values (a one-row call each) are averaged."""
        keep = om > 0
        iv = np.stack([np.minimum(lo, hi), np.maximum(lo, hi)], 1)  # MAPIE does not un-cross coverage; Winkler does
        out = []
        per = lambda f: np.array([f(i) for i in np.flatnonzero(keep)])
        cov = per(lambda i: regression_coverage_score(y[i : i + 1], iv[i : i + 1])[0])
        wid = per(lambda i: regression_mean_width_score(iv[i : i + 1, :, None])[0])
        wk = per(lambda i: regression_mwi_score(y[i : i + 1], np.stack([lo, hi], 1)[i : i + 1, :, None], 1 - alpha))
        for v in (cov, wid, wk):
            out.append(np.sum(om[keep] * v) / np.sum(om[keep]))
        out.append(mean_absolute_error(y, pred, sample_weight=om))
        for q in quants:
            out.append(mean_pinball_loss(y, pred, alpha=q, sample_weight=om))
        return np.array(out)

    rows = []
    for case, hl, wmode in (("lam=1, unit w", np.inf, 0), ("lam=1, w in [.5,2]", np.inf, 1),
                            ("half_life 60, w in [.5,2], 12% zero-weight rows", 60.0, 2)):
        g = np.zeros(len(names))
        for sd in range(min(seeds, 6)):
            n = 250
            y, pred, lo, hi, p, yb = make_rows(sd, n)
            _, _, dt, w = stream(sd, n, 1)
            if wmode == 0:
                w = np.ones(n)
            if wmode == 2:
                w[3::8] = 0.0
            sc = EwScores(len(names))
            om = np.zeros(n)
            for i in range(n):
                lam = lam_of(dt[i], hl)
                om[:i] *= lam
                om[i] = w[i]
                sc.update(row_scores(y[i], lo[i], hi[i], alpha, pred[i], quants), lam, w[i])
            ov = oracle_vec(y, pred, lo, hi, om)
            g = np.maximum(g, np.abs(sc.m - ov) / np.maximum(np.abs(ov), 1e-300))
        raw = np.sum(om[keep := om > 0] * np.array([regression_coverage_score(y[i : i + 1], np.stack([lo, hi], 1)[i : i + 1])[0]
                                                   for i in np.flatnonzero(keep)])) / np.sum(om[keep])
        rows.append([case, *g, abs(sc.m[0] - raw)])
    table("513a: EW scores against the corrected oracles (largest relative gap over seeds, 5% of the rows have crossed bounds, un-crossed for MAPIE's coverage and width as for Winkler; MAPIE for coverage, width, Winkler [y_pis, 1 - confidence_level]; scikit-learn for MAE and pinball [sample_weight])",
          ["case", *names, "coverage vs MAPIE on the raw (crossed) bounds, abs"], rows)

    # log loss with the float64-eps clip
    rows = []
    for case, hl, wmode in (("lam=1, unit w", np.inf, 0), ("half_life 60, zero-weight rows", 60.0, 2)):
        g = 0.0
        for sd in range(seeds):
            n = 300
            y, pred, lo, hi, p, yb = make_rows(sd, n)
            _, _, dt, w = stream(sd, n, 1)
            if wmode == 0:
                w = np.ones(n)
            else:
                w[3::8] = 0.0
            sc = EwScores(1)
            om = np.zeros(n)
            for i in range(n):
                lam = lam_of(dt[i], hl)
                om[:i] *= lam
                om[i] = w[i]
                pc = min(max(p[i], EPS), 1 - EPS)
                sc.update(np.array([-(yb[i] * np.log(pc) + (1 - yb[i]) * np.log(1 - pc))]), lam, w[i])
            ov = log_loss(yb, p, sample_weight=om, labels=[0, 1])
            g = max(g, abs(sc.m[0] - ov) / ov)
        rows.append([case, g])
    table("513b: streaming log loss vs scikit-learn log_loss(y, y_proba, sample_weight), probabilities include exactly 1.0 (clip at float64 eps; relative gap)",
          ["case", "gap"], rows)

    # SSC and D2 over a window
    rows = []
    for wn in (50, 200):
        gs = gd = 0.0
        gt = []
        for sd in range(seeds):
            n = 600
            y, pred, lo, hi, p, yb = make_rows(sd, n)
            win = Window(wn, 0.9)
            for i in range(n):
                l, u = min(lo[i], hi[i]), max(lo[i], hi[i])
                win.push(u - l, 1.0 if l <= y[i] <= u else 0.0, y[i], pinball(y[i] - pred[i], 0.9))
            iv = np.stack([np.minimum(lo, hi), np.maximum(lo, hi)], 1)[-wn:]
            ref = regression_ssc(y[-wn:], iv, num_bins=3)[0]
            gs = max(gs, float(np.max(np.abs(win.ssc() - ref))))
            gd = max(gd, abs(win.d2() - d2_pinball_score(y[-wn:], pred[-wn:], alpha=0.9)))
        rows.append([wn, gs, gd, win.state_bytes()])
    table("513c: window-exact size-stratified coverage (3 bins, absolute gap) and D2 pinball (absolute gap) vs MAPIE regression_ssc and scikit-learn d2_pinball_score; state bytes at that window",
          ["window", "SSC", "D2", "bytes"], rows)

    # D2 at a running quantile (no window): how far from the hindsight null
    rows = []
    for stat in (True, False):
        for rate in (0.02, 0.005):
            for n in (2000, 20000):
                gaps = []
                for sd in range(min(seeds, 10)):
                    y, pred, *_ = make_rows(sd, n, stationary=stat)
                    a = 0.9
                    q, scale = y[0], 1.0
                    num = den = 0.0
                    for i in range(1, n):
                        num += pinball(y[i] - pred[i], a)
                        den += pinball(y[i] - q, a)
                        scale += 0.05 * (abs(y[i] - q) - scale)
                        q += rate * scale * (a - (1.0 if y[i] < q else 0.0))
                    gaps.append((1 - num / den) - d2_pinball_score(y[1:], pred[1:], alpha=0.9))
                rows.append(["stationary" if stat else "random-walk mean", rate, n, float(np.mean(gaps)), float(np.max(np.abs(gaps)))])
    table("513d: D2 pinball (alpha 0.9) with a running-quantile null (8 bytes, step = rate x EW mean abs deviation) minus scikit-learn's hindsight D2 over the whole stream",
          ["stream", "rate", "n", "mean gap", "max |gap|"], rows)

    rows = [[len(names), EwScores(len(names)).state_bytes(), 8 * len(names) + 8]]
    table("513e: state of the mean scores (shared weight + one f64 each); per row of window: SSC ring 16 bytes, D2 ring 16, one shared order index 8 (40 together); independent of the stream",
          ["scores", "bytes", "8 n + 8"], rows)
