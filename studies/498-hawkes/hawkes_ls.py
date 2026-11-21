# ruff: noqa: E501  (a study script: its report lines are long)
"""Study 498, part 1: exact online least-squares fit of a Hawkes process with
sum-of-exponential kernels and given decays, against tick.

Run (from a venv with numpy, tick 0.8.0.2, and numpydoc and six, which tick
imports but does not declare):

    python studies/498-hawkes/hawkes_ls.py

Everything is generated here from seeded numpy generators; no data files.

Parametrisation (tick's): for nodes i, j = 0..D-1 and decays beta_u, u = 0..U-1,

    lambda_i(t) = mu_i + sum_{j,u} alpha_iju * R_ju(t),
    R_ju(t)     = sum_{t_k^j < t} beta_u exp(-beta_u (t - t_k^j)).

Feature vector phi(t) = (1, R_00, ..., R_{D-1,U-1}), length p = 1 + D*U.
theta_i = (mu_i, alpha_i..), so lambda_i(t) = theta_i . phi(t).

Least-squares contrast (tick's docstring):
    L(theta) = sum_i [ int_0^T lambda_i^2 dt - 2 int_0^T lambda_i dN_i ]
             = sum_i [ theta_i' G theta_i - 2 theta_i' b_i ]
with G = int_0^T phi phi' dt (one Gram for all nodes) and
b_i = sum over node-i events of phi(t_k^i -) (left limit: the event's own jump excluded).
"""

from __future__ import annotations

import time

import numpy as np

# ---------------------------------------------------------------- simulation


def simulate(mu, alpha, decays, T, seed):
    """Ogata thinning with numpy's seeded generator. alpha shape (D, D, U)."""
    rng = np.random.default_rng(seed)
    D, U = len(mu), len(decays)
    R = np.zeros((D, U))
    t = 0.0
    events = [[] for _ in range(D)]
    while True:
        lam = mu + np.einsum("iju,ju->i", alpha, R)
        lam_bar = lam.sum()  # intensity only decays until the next event
        w = rng.exponential(1.0 / lam_bar)
        R = R * np.exp(-decays * w)[None, :]
        t += w
        if t >= T:
            break
        lam_new = mu + np.einsum("iju,ju->i", alpha, R)
        if rng.uniform() * lam_bar <= lam_new.sum():
            i = rng.choice(D, p=lam_new / lam_new.sum())
            events[i].append(t)
            R[i, :] += decays
    return [np.asarray(e) for e in events]


def merge(events):
    times = np.concatenate(events)
    nodes = np.concatenate([np.full(len(e), j) for j, e in enumerate(events)])
    order = np.argsort(times, kind="stable")
    return times[order], nodes[order]


# ---------------------------------------------------------------- online, form A


class OnlineInterval:
    """Integrate phi phi' over each gap in closed form. O(p^2) per event.

    Between events R_a(t) = R_a(s) e^{-beta_a (t-s)}, so over a gap of length d:
        int 1        = d
        int R_a      = R_a(s) (1 - e^{-beta_a d}) / beta_a
        int R_a R_b  = R_a(s) R_b(s) (1 - e^{-(beta_a+beta_b) d}) / (beta_a+beta_b)
    (-expm1 for the 1 - e^{-x} factors.)
    """

    def __init__(self, D, decays):
        self.D, self.U = D, len(decays)
        self.beta = np.tile(decays, D)  # feature a = j*U + u has decay beta_u
        p = 1 + D * self.U
        self.bsum = self.beta[:, None] + self.beta[None, :]
        self.G = np.zeros((p, p))
        self.b = np.zeros((D, p))
        self.R = np.zeros(D * self.U)
        self.t = 0.0

    def _advance(self, t):
        d = t - self.t
        R = self.R
        self.G[0, 0] += d
        i1 = -np.expm1(-self.beta * d) / self.beta * R
        self.G[0, 1:] += i1
        self.G[1:, 0] += i1
        self.G[1:, 1:] += np.outer(R, R) * (-np.expm1(-self.bsum * d) / self.bsum)
        self.R = R * np.exp(-self.beta * d)
        self.t = t

    def event(self, t, j):
        self._advance(t)
        self.b[j, 0] += 1.0
        self.b[j, 1:] += self.R  # phi(t-): read before the jump
        U = self.U
        self.R[j * U:(j + 1) * U] += self.beta[j * U:(j + 1) * U]

    def stats(self, T):
        """G, b at the read time T (a copy: reading does not move the state)."""
        snap = OnlineInterval.__new__(OnlineInterval)
        snap.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                         for k, v in self.__dict__.items()}
        snap._advance(T)
        return snap.G, snap.b


# ---------------------------------------------------------------- online, form B


class OnlineJump:
    """The same G from jump sums only. O(U * p) per event, U exps per event.

    Between events d/dt (R_a R_b) = -(beta_a+beta_b) R_a R_b and d/dt R_a = -beta_a R_a,
    so integrating from 0 (all R = 0) to T:
        int_0^T R_a R_b dt = (J_ab - R_a(T) R_b(T)) / (beta_a + beta_b),
        int_0^T R_a dt     = (beta_a N_a - R_a(T)) / beta_a = N_a - R_a(T)/beta_a,
    where J_ab is the sum over events of the jump in R_a R_b. An event of node j
    jumps R_a by beta_a for the U features a of node j only, so
        J_ab += beta_a 1[a in j] R_b + beta_b 1[b in j] R_a + beta_a beta_b 1[a,b in j]
    (R before the jump) touches U rows/columns of J: O(U * D * U) work.
    State: J (DU x DU, symmetric), b (D x p), R (DU), counts N (D), last time t.
    """

    def __init__(self, D, decays):
        self.D, self.U = D, len(decays)
        self.dec = np.asarray(decays, float)
        self.beta = np.tile(decays, D)
        k = D * self.U
        self.J = np.zeros((k, k))
        self.b = np.zeros((D, 1 + k))
        self.R = np.zeros(k)
        self.N = np.zeros(D)
        self.t = 0.0

    def event(self, t, j):
        U = self.U
        self.R *= np.tile(np.exp(-self.dec * (t - self.t)), self.D)  # U exps
        self.t = t
        self.b[j, 0] += 1.0
        self.b[j, 1:] += self.R
        s = slice(j * U, (j + 1) * U)
        bj = self.dec
        row = bj[:, None] * self.R[None, :]  # beta_a R_b, a in j
        row[:, s] += np.outer(bj, bj)  # + beta_a beta_b for a, b in j
        self.J[s, :] += row
        self.J[:, s] += row.T
        self.J[s, s] -= np.outer(bj, bj)  # counted twice above; once is right
        self.R[s] += bj
        self.N[j] += 1.0

    def stats(self, T):
        RT = self.R * np.tile(np.exp(-self.dec * (T - self.t)), self.D)
        k = len(RT)
        G = np.empty((1 + k, 1 + k))
        G[0, 0] = T
        g1 = np.repeat(self.N, self.U) - RT / self.beta
        G[0, 1:] = g1
        G[1:, 0] = g1
        G[1:, 1:] = (self.J - np.outer(RT, RT)) / (self.beta[:, None] + self.beta[None, :])
        return G, self.b.copy()


# ---------------------------------------------------------------- batch


def batch_stats(events, decays, T):
    """G and b from the definitions as double sums over pairs of events (O(n^2)).

    int_0^T R_a R_b dt = beta_a beta_b sum_{k in a's node, l in b's node}
        int_{max(t_k,t_l)}^T e^{-beta_a (t-t_k)} e^{-beta_b (t-t_l)} dt
      = beta_a beta_b sum_{k,l} e^{-beta_a (m-t_k)} e^{-beta_b (m-t_l)}
                                (1 - e^{-(beta_a+beta_b)(T-m)}) / (beta_a+beta_b),
    m = max(t_k, t_l).  int_0^T R_a dt = sum_k (1 - e^{-beta_a (T - t_k)}).
    b_i[a] = sum_{events s of i} sum_{t_k of a's node, t_k < s} beta_a e^{-beta_a (s-t_k)}.
    """
    D, U = len(events), len(decays)
    k = D * U
    G = np.zeros((1 + k, 1 + k))
    b = np.zeros((D, 1 + k))
    G[0, 0] = T
    feats = [(j, u) for j in range(D) for u in range(U)]
    for a, (ja, ua) in enumerate(feats):
        ba, ta = decays[ua], events[ja]
        G[0, 1 + a] = G[1 + a, 0] = np.sum(-np.expm1(-ba * (T - ta)))
        for c, (jc, uc) in enumerate(feats):
            bc, tc = decays[uc], events[jc]
            m = np.maximum(ta[:, None], tc[None, :])
            s = ba + bc
            G[1 + a, 1 + c] = ba * bc * np.sum(
                np.exp(-ba * (m - ta[:, None]) - bc * (m - tc[None, :]))
                * (-np.expm1(-s * (T - m))) / s)
        for i in range(D):
            si = events[i]
            dt = si[:, None] - ta[None, :]
            b[i, 1 + a] = np.sum(np.where(dt > 0, ba * np.exp(-ba * np.where(dt > 0, dt, 0.0)), 0.0))
    for i in range(D):
        b[i, 0] = len(events[i])
    return G, b


# ---------------------------------------------------------------- fit


def loss(G, b, theta):
    """theta shape (D, p): rows (mu_i, alpha_i..)."""
    return float(np.einsum("ip,pq,iq->", theta, G, theta) - 2.0 * np.sum(theta * b))


def minimiser(G, b):
    return np.linalg.solve(G, b.T).T  # theta_i = G^{-1} b_i, one Gram for every node


def minimiser_nonneg(G, b):
    """theta_i >= 0 (tick's learners project onto it): NNLS on the Cholesky
    factor, G = L L', theta'G theta - 2 theta'b = |L'theta - L^{-1}b|^2 - const."""
    from scipy.optimize import nnls

    L = np.linalg.cholesky(G)
    y = np.linalg.solve(L, b.T)
    return np.array([nnls(L.T, y[:, i], maxiter=1000)[0] for i in range(b.shape[0])])


def theta_to_tick(theta, D, U):
    """tick's layout: [mu_0..mu_{D-1}, adjacency (D, D, U) raveled]."""
    return np.concatenate([theta[:, 0], theta[:, 1:].reshape(D, D, U).ravel()])


def tick_to_theta(c, D, U):
    mu = c[:D]
    adj = c[D:].reshape(D, D * U)
    return np.column_stack([mu, adj])


# ---------------------------------------------------------------- main


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


CASES = [
    # name, mu, alpha (D, D, U), decays, T, seed
    ("1d-U2", [0.5], [[[0.3, 0.2]]], [1.0, 5.0], 4000.0, 11),
    ("1d-U3", [0.8], [[[0.2, 0.15, 0.1]]], [0.5, 3.0, 20.0], 3000.0, 12),
    ("2d-U2", [0.4, 0.6], [[[0.2, 0.1], [0.05, 0.1]], [[0.1, 0.05], [0.25, 0.1]]],
     [1.0, 8.0], 3000.0, 13),
    ("2d-U3", [0.3, 0.5], [[[0.1, 0.1, 0.05], [0.05, 0.05, 0.1]],
                           [[0.1, 0.05, 0.05], [0.15, 0.05, 0.1]]],
     [0.7, 4.0, 15.0], 2500.0, 14),
]


def main():
    import warnings

    warnings.filterwarnings("ignore")
    from tick.hawkes import HawkesSumExpKern, ModelHawkesSumExpKernLeastSq

    rows = []
    for name, mu, alpha, decays, T, seed in CASES:
        mu, alpha, decays = np.array(mu), np.array(alpha), np.array(decays, float)
        D, U = len(mu), len(decays)
        p = 1 + D * U
        events = simulate(mu, alpha, decays, T, seed)
        times, nodes = merge(events)
        n = len(times)

        A, B = OnlineInterval(D, decays), OnlineJump(D, decays)
        t0 = time.perf_counter()
        for t, j in zip(times, nodes, strict=True):
            A.event(t, j)
        per_event_us_A = (time.perf_counter() - t0) / n * 1e6
        t0 = time.perf_counter()
        for t, j in zip(times, nodes, strict=True):
            B.event(t, j)
        per_event_us = (time.perf_counter() - t0) / n * 1e6
        GA, bA = A.stats(T)
        GB, bB = B.stats(T)
        Gc, bc = batch_stats(events, decays, T)

        thA, thB, thC = minimiser(GA, bA), minimiser(GB, bB), minimiser(Gc, bc)

        # tick: the model's loss at the same coefficients, and its exact minimiser
        model = ModelHawkesSumExpKernLeastSq(decays).fit(events, end_times=T)
        rng = np.random.default_rng(seed + 100)
        lrel = []
        scale = None
        for _ in range(5):
            th = rng.uniform(0.0, 0.5, size=(D, p))
            ours = loss(GB, bB, th)
            theirs = model.loss(theta_to_tick(th, D, U))
            scale = theirs / ours if scale is None else scale
            lrel.append(abs(theirs - ours / model.n_jumps) / abs(theirs))
        # minimiser of tick's quadratic, from tick's own gradient: the loss is
        # quadratic, so H e_k = grad(e_k) - grad(0) exactly (tick has no hessian
        # for the sum-exp model), and the minimiser is -H^{-1} grad(0).
        nc = D + D * D * U
        g0 = model.grad(np.zeros(nc))
        H = np.column_stack([model.grad(np.eye(nc)[c]) - g0 for c in range(nc)])
        x_tick = -np.linalg.solve(0.5 * (H + H.T), g0)
        th_tick = tick_to_theta(x_tick, D, U)
        # and tick's learner, no penalty (it projects onto theta >= 0)
        lr = HawkesSumExpKern(decays, penalty="none", solver="agd", tol=1e-13,
                              max_iter=20000)
        lr.fit(events)  # the learner takes no end time: it ends at the last event
        th_learn = np.column_stack([lr.baseline, lr.adjacency.reshape(D, D * U)])
        th_last = minimiser_nonneg(*B.stats(times[-1]))  # our fit, read at that time

        rows.append(dict(
            name=name, n=n, D=D, U=U,
            G_AB=rel(GA, GB), b_AB=rel(bA, bB), G_batch=rel(GB, Gc), b_batch=rel(bB, bc),
            th_AB=rel(thA, thB), th_batch=rel(thB, thC),
            loss_vs_tick=max(lrel), tick_scale_x_n=scale * model.n_jumps,
            th_vs_tick_abs=float(np.max(np.abs(thB - th_tick))),
            th_vs_tick_rel=rel(thB, th_tick),
            min_vs_tick=abs(loss(GB, bB, thB) / n - model.loss(x_tick)) / abs(model.loss(x_tick)),
            th_vs_learner_abs=float(np.max(np.abs(th_last - th_learn))),
            min_theta=float(thB.min()),
            us=per_event_us, usA=per_event_us_A, bytes_state=8 * ((D * U) * (D * U + 1) // 2 + D * p + D * U + D + 1),
            theta=thB,
        ))

    for r in rows:
        print(f"\n== {r['name']}: n={r['n']} events, D={r['D']}, U={r['U']}")
        print(f"  online A vs B   : G rel {r['G_AB']:.2e}  b rel {r['b_AB']:.2e}  theta rel {r['th_AB']:.2e}")
        print(f"  online vs batch : G rel {r['G_batch']:.2e}  b rel {r['b_batch']:.2e}  theta rel {r['th_batch']:.2e}")
        print(f"  tick loss       : tick = ours / n_jumps (scale*n = {r['tick_scale_x_n']:.12f}); max rel diff {r['loss_vs_tick']:.2e}")
        print(f"  tick minimiser  : max abs {r['th_vs_tick_abs']:.2e}  rel {r['th_vs_tick_rel']:.2e}; min loss rel {r['min_vs_tick']:.2e}")
        print(f"  tick learner    : vs our non-negative fit at the last event: max abs {r['th_vs_learner_abs']:.2e}"
              f" (unconstrained min theta {r['min_theta']:.3f})")
        print(f"  theta (mu, alpha..) per node:\n{np.array2string(r['theta'], precision=4)}")
        print(f"  per event       : B {r['us']:.1f} us, A {r['usA']:.1f} us (numpy, Python loop); state B {r['bytes_state']} B")


if __name__ == "__main__":
    main()
