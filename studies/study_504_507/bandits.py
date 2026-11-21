"""Task 504: contextual bandits as a policy over per-arm ridge states.

The rheon form keeps, per arm ``a``, the state ``rls`` keeps (square-root
information form)::

    A_a = R_a' R_a         A_a <- lam A_a + w x x'   (the chosen arm)
    u_a = R_a^-T b_a       b_a <- lam b_a + w y x    A_0 = l2 I, b_0 = 0
    beta_a = R_a^-1 u_a    (back-substitution)

with ``R`` upper triangular, kept with a non-negative diagonal (a Givens
rotation's radius is ``hypot``), and folded in by ``d`` rotations of
``[R u; sqrt(w) x' sqrt(w) y]``. ``lam`` decays every arm on every learned row
(the group's clock); ``lam = 1`` is the libraries' undecayed ridge.

Scores, all before the row's reward is learned (rule 2):

    LinUCB (Li et al. 2010)          x'beta_a + alpha * ||R_a^-T x||        (= sqrt(x' A^-1 x))
    river's LinUCBDisjoint           x'beta_a + sqrt(1/noise_prec + x' A^-1 x)
    LinTS (Agrawal and Goyal 2013)   x'(beta_a + M_a z_a),  M_a M_a' = alpha^2 A_a^-1
        compatible square root       M = chol(alpha^2 A^-1)    (mabwiser's, numpy's 'cholesky')
        fast square root             M = alpha R^-1           (one back-substitution, O(d^2))
    epsilon-greedy                   with probability eps a uniform arm, else argmax x'beta_a

Draws, per row, keyed on ``(seed, group, n)`` with ``n`` the policy's learned
rows: lane 0 holds uniforms (index 0 the explore flag, 1..K the random arm's
scores, as mabwiser draws them; or the tie draws of river's rule), lane 1
normals (index ``a*d + i``: arm-major, as mabwiser draws them).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular

import philox as P

LANE_UNIFORM = 0
LANE_NORMAL = 1


# --- the per-row draw source and the adapters that hand it to the libraries ---------


class RowDraws:
    """The draws of the current row: two sequential cursors (uniforms, normals)."""

    def __init__(self, stream: P.DrawStream, max_u: int = 64, max_z: int = 0) -> None:
        self.stream = stream
        self.max_u = max_u
        self.max_z = max_z
        self.set_row(0)

    def set_row(self, n: int) -> None:
        self.n = n
        self.u = self.stream.uniforms(n, self.max_u, lane=LANE_UNIFORM)
        self.z = self.stream.normals(n, self.max_z, lane=LANE_NORMAL) if self.max_z else np.zeros(0)
        self.iu = 0
        self.iz = 0

    def next_u(self, k: int) -> np.ndarray:
        out = self.u[self.iu : self.iu + k]
        assert out.size == k, "row ran out of uniforms"
        self.iu += k
        return out

    def next_z(self, k: int) -> np.ndarray:
        out = self.z[self.iz : self.iz + k]
        assert out.size == k, "row ran out of normals"
        self.iz += k
        return out


class MabGenerator:
    """Stands in for the ``numpy.random.Generator`` inside mabwiser's ``_NumpyRNG``:
    ``random`` serves lane 0, ``multivariate_normal`` serves lane 1 through numpy's
    own 'cholesky' transform (``mean + z @ chol(cov).T``; checked to the bit
    against ``Generator.multivariate_normal`` on the same normals)."""

    def __init__(self, draws: RowDraws) -> None:
        self.d = draws

    def random(self, size=None):
        if size is None:
            return float(self.d.next_u(1)[0])
        shape = (size,) if isinstance(size, int) else tuple(size)
        k = int(np.prod(shape))
        return self.d.next_u(k).reshape(shape)

    def multivariate_normal(self, mean, cov, size=None, method="cholesky"):
        mean = np.asarray(mean, dtype=float)
        m = 1 if size is None else int(size)
        z = self.d.next_z(m * mean.size).reshape(m, mean.size)
        x = z @ np.linalg.cholesky(cov).T
        x += mean
        return x if size is not None else x[0]


class RiverRandom(random.Random):
    """A ``random.Random`` whose ``random()`` serves lane 0. Overriding ``random``
    alone makes CPython's ``choice`` use ``_randbelow_without_getrandbits``:
    ``floor(r * 2^53) % n`` with ``r`` redrawn while ``r >= (2^53 - 2^53 % n) / 2^53``."""

    def __init__(self, draws: RowDraws) -> None:
        self.d = draws
        super().__init__(0)

    def random(self) -> float:  # type: ignore[override]
        return float(self.d.next_u(1)[0])


def river_choice_index(draws: RowDraws, n: int) -> int:
    """CPython's ``Random.choice`` index without getrandbits, longhand."""
    maxsize = 1 << 53
    rem = maxsize % n
    limit = (maxsize - rem) / maxsize
    r = float(draws.next_u(1)[0])
    while r >= limit:
        r = float(draws.next_u(1)[0])
    return int(r * maxsize) % n


# --- the rheon form ------------------------------------------------------------------


@dataclass
class ArmState:
    R: np.ndarray
    u: np.ndarray

    @classmethod
    def new(cls, d: int, l2: float) -> ArmState:
        return cls(np.sqrt(l2) * np.eye(d), np.zeros(d))

    def decay(self, lam: float) -> None:
        if lam != 1.0:
            s = np.sqrt(lam)
            self.R *= s
            self.u *= s

    def learn(self, x: np.ndarray, y: float, w: float = 1.0) -> None:
        """Givens rotations of [R u; sw x' sw y]; the diagonal stays >= 0."""
        if w <= 0.0:
            return
        sw = np.sqrt(w)
        row = sw * np.asarray(x, dtype=float).copy()
        ry = sw * y
        R, u = self.R, self.u
        d = R.shape[0]
        for i in range(d):
            a, b = R[i, i], row[i]
            if b == 0.0:
                continue
            r = np.hypot(a, b)
            c, s = a / r, b / r
            Ri = R[i, i:].copy()
            R[i, i:] = c * Ri + s * row[i:]
            row[i:] = -s * Ri + c * row[i:]
            ui = u[i]
            u[i] = c * ui + s * ry
            ry = -s * ui + c * ry
            R[i, i] = r
            row[i] = 0.0

    def beta(self) -> np.ndarray:
        return solve_triangular(self.R, self.u, lower=False)

    def width(self, x: np.ndarray) -> float:
        """sqrt(x' A^-1 x) = ||R^-T x||."""
        v = solve_triangular(self.R, x, trans="T", lower=False)
        return float(np.sqrt(v @ v))

    def a_inv(self) -> np.ndarray:
        Ri = solve_triangular(self.R, np.eye(self.R.shape[0]), lower=False)
        return Ri @ Ri.T

    def nbytes(self) -> int:
        d = self.R.shape[0]
        return 8 * (d * (d + 1) // 2 + d)


@dataclass
class RheonBandit:
    """policy: 'linucb', 'linucb_pred' (river's bound), 'lints', 'egreedy'.
    ts_root: 'compatible' (chol(alpha^2 A^-1)) or 'fast' (alpha R^-1).
    ties: 'first' (lowest index) or 'river' (river's choice on lane-0 uniforms)."""

    k: int
    d: int
    policy: str
    alpha: float = 1.0
    l2: float = 1.0
    eps: float = 0.1
    lam: float = 1.0
    noise_prec: float = 1.0
    ts_root: str = "compatible"
    ties: str = "first"
    seed: int = P.DEFAULT_SEED
    group: int = 0
    mab_draw_layout: bool = True
    arms: list[ArmState] = field(init=False)
    n: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.arms = [ArmState.new(self.d, self.l2) for _ in range(self.k)]
        self.draws = RowDraws(P.DrawStream(self.seed, self.group), max_u=4 + self.k, max_z=self.k * self.d)

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self.draws.set_row(self.n)
        if self.mab_draw_layout and self.policy in ("linucb", "lints", "egreedy"):
            flag = self.draws.next_u(1)[0]  # mabwiser draws the explore flag for every linear policy
        else:
            flag = 1.0
        if self.policy == "egreedy" and flag < self.eps:
            return self.draws.next_u(self.k).copy()
        out = np.empty(self.k)
        for a, st in enumerate(self.arms):
            b = st.beta()
            if self.policy == "linucb":
                out[a] = x @ b + self.alpha * st.width(x)
            elif self.policy == "linucb_pred":
                w = st.width(x)
                out[a] = x @ b + np.sqrt(1.0 / self.noise_prec + w * w)
            elif self.policy == "lints":
                z = self.draws.next_z(self.d)
                if self.ts_root == "compatible":
                    M = np.linalg.cholesky(self.alpha**2 * st.a_inv())
                    bt = z @ M.T + b
                else:
                    bt = b + self.alpha * solve_triangular(st.R, z, lower=False)
                out[a] = float(np.sum(x * bt))
            elif self.policy == "egreedy":
                out[a] = x @ b
            else:
                raise ValueError(self.policy)
        return out

    def choose(self, s: np.ndarray) -> int:
        if self.ties == "first":
            return int(np.argmax(s))
        top = s.max()
        cand = [i for i in range(self.k) if s[i] == top]
        self.tie_rows = getattr(self, "tie_rows", 0) + (len(cand) > 1)
        return cand[river_choice_index(self.draws, len(cand))] if len(cand) > 1 else cand[0]

    def act(self, x: np.ndarray) -> tuple[int, np.ndarray]:
        s = self.scores(x)
        return self.choose(s), s

    def learn(self, arm: int, x: np.ndarray, y: float, w: float = 1.0) -> None:
        if w <= 0.0:  # rule 9: no decay of the fit's weight beyond the clock's, no draw advance
            return
        for st in self.arms:
            st.decay(self.lam)
        self.arms[arm].learn(x, y, w)
        self.n += 1

    def state_bytes(self) -> int:
        return sum(a.nbytes() for a in self.arms) + 8  # + the learned-row count


# --- the longhand, from the papers ------------------------------------------------------


class Longhand:
    """Li et al. (2010) Algorithm 1 and Agrawal and Goyal (2013), per arm, with the
    sums kept as written (A, b) and every solve and inverse taken afresh."""

    def __init__(self, k, d, policy, alpha=1.0, l2=1.0, eps=0.1, lam=1.0, noise_prec=1.0, ts_root="compatible",
                 ties="first", seed=P.DEFAULT_SEED, group=0, mab_draw_layout=True) -> None:
        self.k, self.d, self.policy = k, d, policy
        self.alpha, self.l2, self.eps, self.lam, self.noise_prec = alpha, l2, eps, lam, noise_prec
        self.ts_root, self.ties, self.mab = ts_root, ties, mab_draw_layout
        self.A = [l2 * np.eye(d) for _ in range(k)]
        self.b = [np.zeros(d) for _ in range(k)]
        self.n = 0
        self.draws = RowDraws(P.DrawStream(seed, group), max_u=4 + k, max_z=k * d)

    def act(self, x):
        x = np.asarray(x, dtype=float)
        self.draws.set_row(self.n)
        flag = self.draws.next_u(1)[0] if (self.mab and self.policy in ("linucb", "lints", "egreedy")) else 1.0
        if self.policy == "egreedy" and flag < self.eps:
            s = self.draws.next_u(self.k).copy()
            return int(np.argmax(s)), s
        s = np.empty(self.k)
        for a in range(self.k):
            Ainv = np.linalg.inv(self.A[a])
            theta = np.linalg.solve(self.A[a], self.b[a])
            if self.policy == "linucb":
                s[a] = theta @ x + self.alpha * np.sqrt(x @ Ainv @ x)
            elif self.policy == "linucb_pred":
                s[a] = theta @ x + np.sqrt(1.0 / self.noise_prec + x @ Ainv @ x)
            elif self.policy == "lints":
                z = self.draws.next_z(self.d)
                if self.ts_root == "compatible":
                    mu = theta + np.linalg.cholesky(self.alpha**2 * Ainv) @ z
                else:  # B = L L', so B^-1 = L^-T L^-1 and L^-T z has covariance B^-1
                    L = np.linalg.cholesky(self.A[a])
                    mu = theta + self.alpha * np.linalg.solve(L.T, z)
                s[a] = mu @ x
            elif self.policy == "egreedy":
                s[a] = theta @ x
        if self.ties == "first":
            return int(np.argmax(s)), s
        cand = [i for i in range(self.k) if s[i] == s.max()]
        return (cand[river_choice_index(self.draws, len(cand))] if len(cand) > 1 else cand[0]), s

    def learn(self, arm, x, y, w=1.0):
        if w <= 0.0:
            return
        for a in range(self.k):
            self.A[a] = self.lam * self.A[a]
            self.b[a] = self.lam * self.b[a]
        self.A[arm] = self.A[arm] + w * np.outer(x, x)
        self.b[arm] = self.b[arm] + w * y * np.asarray(x, dtype=float)
        self.n += 1


# --- the libraries, with the study's draws injected ---------------------------------------


class MabPolicy:
    def __init__(self, k, d, policy, alpha=1.0, l2=1.0, eps=0.1, seed=P.DEFAULT_SEED, group=0) -> None:
        from mabwiser.mab import MAB, LearningPolicy

        lp = {
            "linucb": LearningPolicy.LinUCB(alpha=alpha, l2_lambda=l2),
            "lints": LearningPolicy.LinTS(alpha=alpha, l2_lambda=l2),
            "egreedy": LearningPolicy.LinGreedy(epsilon=eps, l2_lambda=l2),
        }[policy]
        self.k, self.d = k, d
        self.mab = MAB(arms=list(range(k)), learning_policy=lp, seed=1)
        self.draws = RowDraws(P.DrawStream(seed, group), max_u=4 + k, max_z=k * d)
        self.n = 0
        self.fitted = False

    def _inject(self) -> None:
        imp = self.mab._imp
        gen = MabGenerator(self.draws)
        imp.rng.rng = gen
        for m in imp.arm_to_model.values():
            m.rng.rng = gen

    def act(self, x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        if not self.fitted:
            # mabwiser predicts only once fitted: a fit on no rows initialises every arm.
            self.mab.fit(decisions=np.array([], dtype=int), rewards=np.array([]), contexts=np.zeros((0, self.d)))
            self.fitted = True
        self._inject()
        self.draws.set_row(self.n)
        e = self.mab.predict_expectations(x)
        s = np.array([e[a] for a in range(self.k)], dtype=float)
        return int(np.argmax(s)), s

    def learn(self, arm, x, y, w=1.0):
        if w <= 0.0:
            return
        self.mab.partial_fit(decisions=np.array([arm]), rewards=np.array([y]), contexts=np.asarray(x).reshape(1, -1))
        self.n += 1


class RiverPolicy:
    def __init__(self, k, d, l2=1.0, noise_prec=1.0, seed=P.DEFAULT_SEED, group=0) -> None:
        from river import bandit

        self.k, self.d = k, d
        self.pol = bandit.LinUCBDisjoint(alpha=l2, beta=noise_prec, seed=0)
        self.draws = RowDraws(P.DrawStream(seed, group), max_u=4 + k)
        self.pol._rng = RiverRandom(self.draws)
        self.n = 0
        self.names = [f"x{i:02d}" for i in range(d)]

    def ctx(self, x):
        return dict(zip(self.names, map(float, x), strict=True))

    def act(self, x):
        self.draws.set_row(self.n)
        c = self.ctx(x)
        s = np.empty(self.k)
        for a in range(self.k):
            dist = self.pol._bayes_lin_regs[a].predict_dist_one(c)
            s[a] = dist.mu + dist.sigma
        arm = self.pol.pull(list(range(self.k)), context=c)
        return int(arm), s

    def learn(self, arm, x, y, w=1.0):
        if w <= 0.0:
            return
        self.pol.update(arm, self.ctx(x), float(y))
        self.n += 1


# --- the simulated problem ----------------------------------------------------------------


@dataclass
class Env:
    k: int
    d: int
    seed: int
    noise: float = 0.5
    drift_at: int | None = None

    def __post_init__(self) -> None:
        g = np.random.default_rng(self.seed)
        self.theta = g.normal(0, 1, (self.k, self.d)) / np.sqrt(self.d)
        self.theta2 = g.normal(0, 1, (self.k, self.d)) / np.sqrt(self.d)
        self.g = g

    def context(self, t: int) -> np.ndarray:
        x = self.g.normal(0, 1, self.d)
        x[0] = 1.0  # an intercept
        return x

    def means(self, t: int, x: np.ndarray) -> np.ndarray:
        th = self.theta2 if (self.drift_at is not None and t >= self.drift_at) else self.theta
        return th @ x


def simulate(policy, env: Env, T: int, weights: np.ndarray | None = None) -> dict[str, object]:
    """Closed loop: the policy acts on the context, the environment returns the
    chosen arm's reward, the policy learns it. Regret is the pseudo-regret
    ``max_a mu_a(x) - mu_chosen(x)``, summed."""
    reg = np.empty(T)
    arms = np.empty(T, dtype=int)
    scores = []
    for t in range(T):
        x = env.context(t)
        a, s = policy.act(x)
        mu = env.means(t, x)
        y = mu[a] + env.noise * env.g.normal()
        reg[t] = mu.max() - mu[a]
        arms[t] = a
        scores.append(s)
        policy.learn(a, x, y, 1.0 if weights is None else float(weights[t]))
    return {"regret": np.cumsum(reg), "arms": arms, "scores": np.array(scores)}


def replay(policies: dict[str, object], env: Env, T: int, driver: str) -> dict[str, dict[str, float]]:
    """Every policy scores the same row; the ``driver``'s choice is the logged
    action and every policy learns it. Choices and scores compared row by row."""
    ch = {n: np.empty(T, dtype=int) for n in policies}
    sc = {n: np.empty((T, env.k)) for n in policies}
    for t in range(T):
        x = env.context(t)
        for n, p in policies.items():
            ch[n][t], sc[n][t] = p.act(x)
        a = ch[driver][t]
        y = env.means(t, x)[a] + env.noise * env.g.normal()
        for p in policies.values():
            p.learn(a, x, y)
    out = {}
    ref = sc[driver]
    for n in policies:
        if n == driver:
            continue
        diff = np.abs(sc[n] - ref)
        rel = diff / np.maximum(np.abs(ref), 1e-300)
        out[n] = {
            "rows differing in choice": int(np.sum(ch[n] != ch[driver])),
            "max |score diff|": float(diff.max()),
            "max relative": float(rel.max()),
        }
    return out


class RandomPolicy:
    """The baseline: a uniform arm from lane 0 by multiply-shift (integer only)."""

    def __init__(self, k, seed=P.DEFAULT_SEED, group=0):
        self.k, self.n, self.s = k, 0, P.DrawStream(seed, group)

    def act(self, x):
        return P.below(self.s.words(self.n, 1)[0], self.k), np.zeros(self.k)

    def learn(self, arm, x, y, w=1.0):
        if w > 0:
            self.n += 1


class MabOwnRng(MabPolicy):
    """mabwiser with its own numpy generator (nothing injected): a second opinion
    on regret, not on draws."""

    def _inject(self) -> None:
        pass


class VWPolicy:
    """Vowpal Wabbit's ``--cb_explore`` (epsilon-greedy over its own hashed SGD
    learner): a regret band only. The action is drawn from VW's pmf with lane 0."""

    def __init__(self, k, d, eps=0.1, seed=P.DEFAULT_SEED, group=0, extra=""):
        from vowpalwabbit import Workspace

        self.vw = Workspace(f"--cb_explore {k} --epsilon {eps} --quiet --random_seed 1 {extra}".strip())
        self.k, self.n, self.s = k, 0, P.DrawStream(seed, group)
        self.last = None

    def feat(self, x):
        return "| " + " ".join(f"x{i}:{v:.17g}" for i, v in enumerate(x))

    def act(self, x):
        pmf = np.asarray(self.vw.predict(self.feat(x)), dtype=float)
        u = P.u01(self.s.words(self.n, 1)[0])
        a = int(min(np.searchsorted(np.cumsum(pmf), u * pmf.sum(), side="right"), self.k - 1))
        self.last = pmf[a]
        return a, pmf

    def learn(self, arm, x, y, w=1.0):
        if w <= 0:
            return
        self.vw.learn(f"{arm + 1}:{-y:.17g}:{self.last:.17g} {self.feat(x)}")
        self.n += 1


def fast_root_covariance(policy: RheonBandit, ref: Longhand) -> float:
    """Max relative error of (alpha R^-1)(alpha R^-1)' against alpha^2 A^-1 over the
    arms: the fast square root's covariance, exact up to rounding."""
    worst = 0.0
    for st, A in zip(policy.arms, ref.A, strict=True):
        M = policy.alpha * solve_triangular(st.R, np.eye(policy.d), lower=False)
        target = policy.alpha**2 * np.linalg.inv(A)
        worst = max(worst, float(np.max(np.abs(M @ M.T - target)) / np.max(np.abs(target))))
    return worst


def regret_table(T: int = 2000, seeds: int = 8, k: int = 5, d: int = 6, with_libs: bool = True):
    def mk():
        out = {
            "LinUCB alpha=1 (rheon form)": lambda s: RheonBandit(k, d, "linucb", alpha=1.0, seed=s),
            "LinUCB, river's bound (rheon form)": lambda s: RheonBandit(k, d, "linucb_pred", seed=s),
            "LinTS alpha=1, fast root (rheon form)": lambda s: RheonBandit(k, d, "lints", ts_root="fast", seed=s),
            "LinTS alpha=0.3, fast root (rheon form)": lambda s: RheonBandit(
                k, d, "lints", alpha=0.3, ts_root="fast", seed=s
            ),
            "LinTS alpha=1, compatible root (rheon form)": lambda s: RheonBandit(k, d, "lints", seed=s),
            "eps-greedy 0.1 (rheon form)": lambda s: RheonBandit(k, d, "egreedy", eps=0.1, seed=s),
            "greedy (eps 0)": lambda s: RheonBandit(k, d, "egreedy", eps=0.0, seed=s),
            "uniform random": lambda s: RandomPolicy(k, seed=s),
        }
        if with_libs:
            out["mabwiser LinTS alpha=1, its own generator"] = lambda s: MabOwnRng(k, d, "lints", alpha=1.0)
            out["mabwiser LinUCB alpha=1"] = lambda s: MabPolicy(k, d, "linucb", alpha=1.0, seed=s)
            out["Vowpal Wabbit --cb_explore, eps 0.1"] = lambda s: VWPolicy(k, d, 0.1, seed=s)
        return out

    rows = []
    for name, f in mk().items():
        finals, halves = [], []
        for s in range(seeds):
            pol = f(s)
            if isinstance(pol, MabOwnRng):
                from mabwiser.mab import MAB, LearningPolicy

                pol.mab = MAB(arms=list(range(k)), learning_policy=LearningPolicy.LinTS(alpha=1.0), seed=1000 + s)
            r = simulate(pol, Env(k, d, seed=100 + s), T)["regret"]
            finals.append(r[-1])
            halves.append(r[T // 2 - 1])
        finals = np.array(finals)
        rows.append((name, float(np.mean(halves)), float(finals.mean()), float(finals.std(ddof=1))))
    return rows


def drift_table(T: int = 2000, seeds: int = 8, k: int = 5, d: int = 6):
    rows = []
    for lam in [1.0, 0.995, 0.98]:
        for pol in ["linucb", "lints"]:
            tot = []
            for s in range(seeds):
                p = RheonBandit(k, d, pol, alpha=1.0, lam=lam, ts_root="fast", seed=s)
                r = simulate(p, Env(k, d, seed=200 + s, drift_at=T // 2), T)["regret"]
                tot.append(r[-1] - r[T // 2 - 1])
            rows.append((pol, lam, float(np.mean(tot)), float(np.std(tot, ddof=1))))
    return rows


# --- chunking, save/resume and zero-weight rows -------------------------------------------


def save_state(p: RheonBandit) -> bytes:
    """The state a build would keep: n, then per arm R's upper triangle and u."""
    import struct

    parts = [struct.pack("<Q", p.n)]
    iu = np.triu_indices(p.d)
    for st in p.arms:
        parts.append(P.f64_bytes(st.R[iu]))
        parts.append(P.f64_bytes(st.u))
    return b"".join(parts)


def load_state(p: RheonBandit, b: bytes) -> RheonBandit:
    import struct

    q = RheonBandit(p.k, p.d, p.policy, alpha=p.alpha, l2=p.l2, eps=p.eps, lam=p.lam, noise_prec=p.noise_prec,
                    ts_root=p.ts_root, ties=p.ties, seed=p.seed, group=p.group, mab_draw_layout=p.mab_draw_layout)
    (q.n,) = struct.unpack_from("<Q", b, 0)
    off = 8
    m = p.d * (p.d + 1) // 2
    iu = np.triu_indices(p.d)
    for st in q.arms:
        R = np.zeros((p.d, p.d))
        R[iu] = np.frombuffer(b, "<f8", m, off)
        off += 8 * m
        st.R = R
        st.u = np.frombuffer(b, "<f8", p.d, off).copy()
        off += 8 * p.d
    assert off == len(b)
    return q


def logged_stream(T: int, k: int, d: int, seed: int):
    env = Env(k, d, seed)
    rows = []
    for t in range(T):
        x = env.context(t)
        a = int(env.g.integers(0, k))
        y = float(env.means(t, x)[a] + env.noise * env.g.normal())
        w = float(env.g.choice([0.0, 1.0, 1.0, 1.0, 0.5]))
        rows.append((x, a, y, w))
    rows[0] = (rows[0][0], rows[0][1], rows[0][2], 0.0)  # a zero-weight first row
    rows[1] = (rows[1][0], rows[1][1], rows[1][2], 0.0)
    return rows


def run_logged(make, rows, chunks, resume: bool) -> str:
    import hashlib

    p = make()
    h = hashlib.sha256()
    i = 0
    for size in chunks:
        for x, a, y, w in rows[i : i + size]:
            act, s = p.act(x)
            h.update(np.int64(act).tobytes() + P.f64_bytes(s))
            p.learn(a, x, y, w)
        i += size
        if resume:
            p = load_state(p, save_state(p))
    return h.hexdigest()[:16]


def bandit_invariance(T: int = 1500, k: int = 4, d: int = 5):
    import streamcheck as S

    rows = logged_stream(T, k, d, 7)
    makers = {
        "LinTS, fast root": lambda: RheonBandit(k, d, "lints", ts_root="fast", seed=3, group=P.group_hash("g")),
        "eps-greedy 0.2": lambda: RheonBandit(k, d, "egreedy", eps=0.2, seed=3, group=P.group_hash("g")),
        "LinUCB, decay 0.99": lambda: RheonBandit(k, d, "linucb", lam=0.99, seed=3),
    }
    out = []
    for name, mk in makers.items():
        digests = set()
        for cname, ch in S.chunkings(T).items():
            for resume in (False, True):
                digests.add(run_logged(mk, rows, ch, resume))
        out.append((name, len(S.chunkings(T)) * 2, len(digests), sorted(digests)[0]))
    return out


def zero_weight_checks(k: int = 3, d: int = 4):
    rows = logged_stream(10, k, d, 11)
    p = RheonBandit(k, d, "lints", ts_root="fast", seed=5)
    x0, a0, y0, _ = rows[0]
    s_before = save_state(p)
    a1, sc1 = p.act(x0)
    p.learn(a0, x0, y0, 0.0)
    unchanged = save_state(p) == s_before
    a2, sc2 = p.act(x0)
    p.learn(a0, x0, y0, 1.0)
    return [
        ("a zero-weight first row leaves the state (R, u, n) unchanged, byte for byte", unchanged),
        ("the next row, same context, gets the same draws and scores", sc1.tobytes() == sc2.tobytes()),
        ("after one learned row n = 1", p.n == 1),
    ]


def state_bytes_table():
    rows = []
    for k in [2, 5, 10, 50]:
        for d in [4, 8, 16, 32]:
            rows.append((k, d, k * 8 * (d * (d + 1) // 2 + d) + 8))
    return rows


def mvn_transform_check() -> bool:
    """numpy's ``multivariate_normal(method='cholesky')`` is ``mean + z @ chol(cov).T``
    on its own standard normals, to the bit (the adapter relies on it)."""
    ok = True
    for s in range(20):
        g = np.random.default_rng(s)
        d = int(g.integers(2, 9))
        B_ = g.normal(size=(d, d))
        cov = B_ @ B_.T + d * np.eye(d)
        mean = g.normal(size=d)
        a = np.random.default_rng(1000 + s).multivariate_normal(mean, cov, size=1, method="cholesky")
        z = np.random.default_rng(1000 + s).standard_normal((1, d))
        b = z @ np.linalg.cholesky(cov).T
        b += mean
        ok &= a.tobytes() == b.tobytes()
    return ok


def oracle_table(T: int = 1000, k: int = 5, d: int = 6, seed: int = 1):
    rows = []

    def add(label, res):
        for n, r in res.items():
            rows.append((label, n, r["rows differing in choice"], r["max |score diff|"], r["max relative"]))

    for pol, lab in [("linucb", "LinUCB alpha=1"), ("lints", "LinTS alpha=1"), ("egreedy", "LinGreedy eps=0.1")]:
        ps = {
            "mabwiser": MabPolicy(k, d, pol, alpha=1.0, eps=0.1),
            "rheon form": RheonBandit(k, d, pol, alpha=1.0, eps=0.1),
            "longhand": Longhand(k, d, pol, alpha=1.0, eps=0.1),
        }
        add(lab + " (driver: mabwiser)", replay(ps, Env(k, d, seed), T, "mabwiser"))
    fast = RheonBandit(k, d, "lints", ts_root="fast")
    lf = Longhand(k, d, "lints", ts_root="fast")
    add("LinTS alpha=1, fast root (driver: longhand)", replay({"longhand": lf, "rheon form": fast}, Env(k, d, seed), T,
                                                             "longhand"))
    cov_err = fast_root_covariance(fast, lf)
    rp = RheonBandit(k, d, "linucb_pred", ties="river")
    ps = {"river": RiverPolicy(k, d), "rheon form": rp, "longhand": Longhand(k, d, "linucb_pred", ties="river")}
    add("river LinUCBDisjoint (driver: river)", replay(ps, Env(k, d, seed), T, "river"))
    return rows, cov_err, getattr(rp, "tie_rows", 0)
