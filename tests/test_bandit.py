"""Contextual linear bandits (`rp.spec.bandit`): LinUCB, LinTS and epsilon-greedy.

What is held, and why:

- **LinUCB and epsilon-greedy linear expectations against ``mabwiser``** at
  ``half_life=float("inf")`` (both with and without an intercept): exact per-arm
  scores to round-off (``<= 1e-12``) and identical greedy arm choices.
- **Exact epsilon-greedy choice probabilities** (``1 - epsilon + epsilon / K``
  on the greedy arm, ``epsilon / K`` otherwise) and empirical exploration rate.
- **LinTS posterior moments**: across 400 independent Philox seeds at a fixed
  learned state, the sampled arm scores recover the exact ridge posterior mean
  ``z' beta_a`` and variance ``alpha^2 z' A_a^-1 z``.
- **Philox4x64-10 draw stream guarantees**: ``predict == step`` to the bit for
  all three policies, zero-weight rows leave state byte-identical and do not
  advance the draw counter, chunk and save/restore invariance, and distinct
  draws across seeds, groups and spec names.
- **Bank contract**: learning convergence, exponential decay forgetting,
  ``predict`` without logged ``action``/``reward``, null reward/action handling,
  ``embargo``, ``coef`` / ``coef_index`` (``arm`` and ``term``), ``describe``,
  ``min_weight`` withholding, and every spec/runtime refusal.
- **Fifty-seed oracle grid** (``bandit.mabwiser``) held against ``mabwiser``.
"""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import polars as pl
import pytest
from mabwiser.mab import MAB, LearningPolicy

import ledger
from rheon import polars as rp

TIER = "mixed"

ARMS = ["a", "b", "c"]


def _save_bytes(bank: rp.ModelBank) -> bytes:
    buf = io.BytesIO()
    bank.save(buf)
    return buf.getvalue()


def _load_bytes(raw: bytes) -> rp.ModelBank:
    return rp.ModelBank.load(io.BytesIO(raw))


def _frame(
    n: int,
    seed: int,
    *,
    arms: list[str] = ARMS,
    k: int = 3,
    weights: bool = False,
) -> pl.DataFrame:
    rng = np.random.default_rng(504_000 + seed)
    X = rng.standard_normal((n, k))
    true_beta = rng.standard_normal((len(arms), k))
    true_bias = rng.standard_normal(len(arms)) * 0.5
    idx = rng.integers(0, len(arms), size=n)
    action_col = [arms[i] for i in idx]
    r = np.array([true_bias[i] + X[row] @ true_beta[i] for row, i in enumerate(idx)])
    r = r + 0.15 * rng.standard_normal(n)
    cols: dict[str, Any] = {
        "t": np.arange(1, n + 1, dtype=float),
        **{f"x{j}": X[:, j] for j in range(k)},
        "action": action_col,
        "r": r,
    }
    if weights:
        cols["w"] = rng.uniform(0.25, 2.0, size=n)
    return pl.DataFrame(cols)


def _mabwiser_scores(
    df: pl.DataFrame,
    *,
    arms: list[str],
    features: list[str],
    policy: str,
    alpha: float = 1.0,
    l2: float = 1.0,
    fit_intercept: bool = True,
) -> dict[str, np.ndarray]:
    """Row-by-row out-of-sample per-arm scores from ``mabwiser.mab.MAB``."""
    X = np.column_stack([df[f].to_numpy() for f in features])
    Z = np.column_stack([np.ones(len(df)), X]) if fit_intercept else X
    lp = (
        LearningPolicy.LinUCB(alpha=alpha, l2_lambda=l2)
        if policy == "linucb"
        else LearningPolicy.LinGreedy(epsilon=0.0, l2_lambda=l2)
    )
    mab = MAB(arms=arms, learning_policy=lp, seed=0)
    # Initialize per-arm models before row 0 so predict_expectations works
    # out-of-sample from the very first row.
    mab._imp.num_features = Z.shape[1]
    for arm in arms:
        mab._imp.arm_to_model[arm].init(num_features=Z.shape[1])
    mab._is_initial_fit = True

    scores = {arm: np.empty(len(df), dtype=float) for arm in arms}
    logged_arms = df["action"].to_list()
    rewards = df["r"].to_numpy()
    for i in range(len(df)):
        ctx = Z[i : i + 1]
        exp = mab.predict_expectations(ctx)
        for arm in arms:
            scores[arm][i] = float(exp[arm])
        mab.partial_fit(decisions=[logged_arms[i]], rewards=[rewards[i]], contexts=ctx)
    return scores


class TestAgainstMabwiser:
    @pytest.mark.parametrize("fit_intercept", [True, False])
    @pytest.mark.parametrize("alpha", [0.0, 0.75, 2.0])
    def test_linucb_matches_mabwiser_row_by_row(self, fit_intercept: bool, alpha: float) -> None:
        df = _frame(100, seed=1)
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            policy="linucb",
            alpha=alpha,
            l2=1.0,
            fit_intercept=fit_intercept,
            half_life=float("inf"),
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("b")
        ref = _mabwiser_scores(
            df,
            arms=ARMS,
            features=["x0", "x1", "x2"],
            policy="linucb",
            alpha=alpha,
            l2=1.0,
            fit_intercept=fit_intercept,
        )
        for a in ARMS:
            diff = np.max(np.abs(out[f"score_{a}"].to_numpy() - ref[a]))
            assert diff <= 1e-12, (fit_intercept, alpha, a, diff)
        # Chosen arm matches argmax of per-arm scores (ties pick lowest arm).
        score_mat = np.column_stack([out[f"score_{a}"].to_numpy() for a in ARMS])
        want_idx = np.argmax(score_mat, axis=1)
        want_arm = [ARMS[i] for i in want_idx]
        assert out["arm"].to_list() == want_arm
        assert np.allclose(out["score"].to_numpy(), score_mat[np.arange(len(df)), want_idx], atol=0)

    def test_linucb_l2_matches_mabwiser_after_every_arm_seen(self) -> None:
        """At ``l2 != 1.0``, mabwiser's ``_RidgeRegression.init`` sets
        ``A_inv = l2 * I`` instead of ``(1 / l2) * I`` until an arm's first
        observation; once every arm has learned once, both agree to round-off."""
        df = _frame(100, seed=2)
        l2 = 2.5
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            policy="linucb",
            alpha=1.1,
            l2=l2,
            half_life=float("inf"),
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("b")
        ref = _mabwiser_scores(
            df,
            arms=ARMS,
            features=["x0", "x1", "x2"],
            policy="linucb",
            alpha=1.1,
            l2=l2,
            fit_intercept=True,
        )
        seen: set[str] = set()
        first_all = None
        for i, a in enumerate(df["action"].to_list()):
            if len(seen) == len(ARMS):
                first_all = i
                break
            seen.add(a)
        assert first_all is not None
        for a in ARMS:
            diff = np.max(np.abs(out[f"score_{a}"].to_numpy()[first_all:] - ref[a][first_all:]))
            assert diff <= 1e-12, (a, diff)

    def test_eps_greedy_scores_match_mabwiser_and_probs_are_exact(self) -> None:
        df = _frame(300, seed=3)
        eps = 0.2
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            policy="eps_greedy",
            epsilon=eps,
            l2=1.0,
            seed=7,
            half_life=float("inf"),
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("b")
        ref = _mabwiser_scores(
            df,
            arms=ARMS,
            features=["x0", "x1", "x2"],
            policy="eps_greedy",
            l2=1.0,
            fit_intercept=True,
        )
        for a in ARMS:
            diff = np.max(np.abs(out[f"score_{a}"].to_numpy() - ref[a]))
            assert diff <= 1e-12, (a, diff)
        score_mat = np.column_stack([out[f"score_{a}"].to_numpy() for a in ARMS])
        greedy_arm = [ARMS[i] for i in np.argmax(score_mat, axis=1)]
        chosen_arm = out["arm"].to_list()
        probs = out["prob"].to_numpy()
        p_greedy = 1.0 - eps + eps / len(ARMS)
        p_other = eps / len(ARMS)
        for i, (c, g, p) in enumerate(zip(chosen_arm, greedy_arm, probs, strict=True)):
            expected = p_greedy if c == g else p_other
            assert abs(p - expected) <= 1e-15, (i, c, g, p, expected)
            assert out["score"][i] == out[f"score_{c}"][i]
        # Empirical non-greedy rate sits near eps * (K - 1) / K.
        non_greedy = sum(c != g for c, g in zip(chosen_arm, greedy_arm, strict=True)) / len(df)
        expected_ng = eps * (len(ARMS) - 1) / len(ARMS)
        assert abs(non_greedy - expected_ng) < 0.08


class TestPoliciesAndPhilox:
    @pytest.mark.parametrize("policy", ["linucb", "lints", "eps_greedy"])
    def test_predict_matches_step_bit_for_bit_on_every_row(self, policy: str) -> None:
        df = _frame(60, seed=4)
        extra = {"epsilon": 0.25} if policy == "eps_greedy" else {"alpha": 1.2}
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            policy=policy,
            seed=99,
            half_life=40.0,
            **extra,
        )
        bank = rp.ModelBank([spec])
        bank.fit_predict(df.slice(0, 1))
        for i in range(1, len(df)):
            row = df.slice(i, 1)
            pred = bank.predict(row.drop("action", "r")).unnest("b")
            step = bank.fit_predict(row).unnest("b")
            assert pred["arm"].to_list() == step["arm"].to_list()
            for col in ["score", *(f"score_{a}" for a in ARMS)]:
                assert np.array_equal(pred[col].to_numpy(), step[col].to_numpy())
            if policy == "eps_greedy":
                assert np.array_equal(pred["prob"].to_numpy(), step["prob"].to_numpy())

    @pytest.mark.parametrize("policy", ["linucb", "lints", "eps_greedy"])
    def test_zero_weight_leaves_state_and_draw_counter_identical(self, policy: str) -> None:
        df = _frame(40, seed=5)
        extra = {"epsilon": 0.3} if policy == "eps_greedy" else {"alpha": 1.0}
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            weight="w",
            policy=policy,
            seed=17,
            half_life=30.0,
            **extra,
        )
        warm = df.slice(0, 25).with_columns(pl.lit(1.0).alias("w"))
        zero = df.slice(25, 5).with_columns(pl.lit(0.0).alias("w"))
        tail = df.slice(30, 10).with_columns(pl.lit(1.0).alias("w"))

        b1 = rp.ModelBank([spec])
        b1.fit_predict(warm)
        before = _save_bytes(b1)
        out_zero = b1.fit_predict(zero).unnest("b")
        # Every zero-weight row matches predict() on the unchanged state and
        # leaves weight_sum and subsequent draws identical to a bank restored
        # before the zero-weight rows.
        pred_zero = b1.predict(zero.drop("action", "r")).unnest("b")
        assert out_zero["arm"].to_list() == pred_zero["arm"].to_list()
        assert np.array_equal(out_zero["score"].to_numpy(), pred_zero["score"].to_numpy())
        assert np.array_equal(out_zero["weight_sum"].to_numpy(), pred_zero["weight_sum"].to_numpy())
        b2 = _load_bytes(before)
        t1 = b1.fit_predict(tail).unnest("b")
        t2 = b2.fit_predict(tail).unnest("b")
        assert t1["arm"].to_list() == t2["arm"].to_list()
        assert np.array_equal(t1["score"].to_numpy(), t2["score"].to_numpy())
        assert np.array_equal(t1["weight_sum"].to_numpy(), t2["weight_sum"].to_numpy())

    @pytest.mark.parametrize("policy", ["linucb", "lints", "eps_greedy"])
    def test_chunk_and_save_restore_invariance(self, policy: str) -> None:
        df = _frame(90, seed=6)
        extra = {"epsilon": 0.2} if policy == "eps_greedy" else {"alpha": 1.0}
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            policy=policy,
            seed=42,
            half_life=50.0,
            **extra,
        )
        one = rp.ModelBank([spec]).fit_predict(df).unnest("b")

        b_chunk = rp.ModelBank([spec])
        parts = []
        for start in range(0, len(df), 7):
            parts.append(b_chunk.fit_predict(df.slice(start, 7)).unnest("b"))
            if start == 28:
                b_chunk = _load_bytes(_save_bytes(b_chunk))
        many = pl.concat(parts)
        assert one["arm"].to_list() == many["arm"].to_list()
        for col in ["score", *(f"score_{a}" for a in ARMS)]:
            assert np.array_equal(one[col].to_numpy(), many[col].to_numpy())

    def test_philox_stream_isolates_seed_group_and_spec_name(self) -> None:
        df = _frame(60, seed=7)
        s0 = rp.spec.bandit(
            "b0",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            policy="lints",
            seed=0,
            half_life=50.0,
        )
        s1 = rp.spec.bandit(
            "b0",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            policy="lints",
            seed=1,
            half_life=50.0,
        )
        s_name = rp.spec.bandit(
            "b_other",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            policy="lints",
            seed=0,
            half_life=50.0,
        )
        o0 = rp.ModelBank([s0]).fit_predict(df).unnest("b0")["score_a"].to_numpy()
        o1 = rp.ModelBank([s1]).fit_predict(df).unnest("b0")["score_a"].to_numpy()
        on = rp.ModelBank([s_name]).fit_predict(df).unnest("b_other")["score_a"].to_numpy()
        assert not np.allclose(o0, o1)
        assert not np.allclose(o0, on)

        # Two groups fed identical rows receive independent Philox keys.
        df_g = pl.concat(
            [df.with_columns(pl.lit("g1").alias("g")), df.with_columns(pl.lit("g2").alias("g"))]
        )
        sg = rp.spec.bandit(
            "bg",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            group="g",
            policy="lints",
            seed=0,
            half_life=50.0,
        )
        og = rp.ModelBank([sg]).fit_predict(df_g).unnest("bg")
        g1 = og.filter(pl.col("g") == "g1")["score_a"].to_numpy()
        g2 = og.filter(pl.col("g") == "g2")["score_a"].to_numpy()
        assert not np.allclose(g1, g2)

    def test_lints_posterior_mean_and_variance_match_ridge_posterior(self) -> None:
        """Over 400 independent Philox seeds at a fixed learned state, LinTS
        draws have mean ``z' beta_a`` and variance ``alpha^2 * z' A_a^-1 z``."""
        df = _frame(90, seed=8)
        alpha = 1.3
        l2 = 1.5
        query = df.slice(89, 1).drop("action", "r")
        train = df.slice(0, 89)

        # Exact ridge posterior at `train` via numpy.
        X = np.column_stack([train[f].to_numpy() for f in ("x0", "x1", "x2")])
        Z = np.column_stack([np.ones(len(train)), X])
        z_q = np.array([1.0, query["x0"][0], query["x1"][0], query["x2"][0]])
        arm_arr = np.array(train["action"].to_list())
        r_arr = train["r"].to_numpy()

        exact_mean = {}
        exact_var = {}
        for a in ARMS:
            mask = arm_arr == a
            Za, ra = Z[mask], r_arr[mask]
            A = l2 * np.eye(Z.shape[1]) + Za.T @ Za
            A_inv = np.linalg.inv(A)
            beta = A_inv @ (Za.T @ ra)
            exact_mean[a] = float(z_q @ beta)
            exact_var[a] = float((alpha**2) * (z_q @ A_inv @ z_q))

        n_seeds = 400
        draws = {a: np.empty(n_seeds) for a in ARMS}
        for seed in range(n_seeds):
            spec = rp.spec.bandit(
                "b",
                arm="action",
                arms=ARMS,
                reward="r",
                features=["x0", "x1", "x2"],
                policy="lints",
                alpha=alpha,
                l2=l2,
                seed=seed,
                half_life=float("inf"),
            )
            bank = rp.ModelBank([spec])
            bank.fit_predict(train)
            pred = bank.predict(query).unnest("b")
            for a in ARMS:
                draws[a][seed] = float(pred[f"score_{a}"][0])

        for a in ARMS:
            se_mean = math.sqrt(exact_var[a] / n_seeds)
            assert abs(draws[a].mean() - exact_mean[a]) < 4.0 * se_mean
            ratio = draws[a].var(ddof=1) / exact_var[a]
            assert 0.80 < ratio < 1.25, (a, ratio)

    @pytest.mark.parametrize("policy", ["linucb", "lints", "eps_greedy"])
    def test_policies_recover_optimal_contextual_arm(self, policy: str) -> None:
        rng = np.random.default_rng(504_99)
        n_train, n_test = 600, 150
        X = rng.standard_normal((n_train + n_test, 2))
        # Arm a prefers +x0, arm b prefers -x0, arm c prefers +x1.
        rewards_all = np.column_stack([2.0 * X[:, 0], -2.0 * X[:, 0], 2.0 * X[:, 1]])
        logged_idx = rng.integers(0, 3, size=n_train)
        train = pl.DataFrame(
            {
                "x0": X[:n_train, 0],
                "x1": X[:n_train, 1],
                "action": [ARMS[i] for i in logged_idx],
                "r": rewards_all[np.arange(n_train), logged_idx]
                + 0.1 * rng.standard_normal(n_train),
            }
        )
        test = pl.DataFrame({"x0": X[n_train:, 0], "x1": X[n_train:, 1]})
        extra = {"epsilon": 0.05} if policy == "eps_greedy" else {"alpha": 0.25}
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1"],
            policy=policy,
            seed=11,
            half_life=float("inf"),
            **extra,
        )
        bank = rp.ModelBank([spec])
        bank.fit_predict(train)
        chosen = bank.predict(test).unnest("b")["arm"].to_list()
        truth = [ARMS[i] for i in np.argmax(rewards_all[n_train:], axis=1)]
        acc = sum(c == t for c, t in zip(chosen, truth, strict=True)) / n_test
        assert acc >= 0.88, (policy, acc)

    def test_decay_forgets_after_arm_reward_swap(self) -> None:
        rng = np.random.default_rng(504_77)
        n_half = 250
        x = rng.standard_normal(2 * n_half)
        logged = ["a" if i % 2 == 0 else "b" for i in range(2 * n_half)]
        r = [
            (2.0 if a == "a" else -2.0) if i < n_half else (-2.0 if a == "a" else 2.0)
            for i, a in enumerate(logged)
        ]
        df = pl.DataFrame({"x0": x, "action": logged, "r": r})
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=["a", "b"],
            reward="r",
            features=["x0"],
            policy="linucb",
            alpha=0.2,
            half_life=25.0,
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("b")
        assert out["arm"].slice(100, 100).to_list().count("a") > 90
        assert out["arm"].tail(80).to_list().count("b") > 75


class TestBankContractAndRefusals:
    def test_null_reward_or_arm_scores_without_learning(self) -> None:
        df = _frame(30, seed=9)
        warm = df.slice(0, 15)
        missing = df.slice(15, 4).with_columns(
            pl.Series("r", [None, 1.0, None, 0.5], dtype=pl.Float64),
            pl.Series("action", ["a", None, None, "b"], dtype=pl.String),
        )
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1", "x2"],
            policy="linucb",
            half_life=float("inf"),
            coef_every=0,
        )
        bank = rp.ModelBank([spec])
        bank.fit_predict(warm)
        out = bank.fit_predict(missing).unnest("b")
        # First 3 rows have missing reward or arm -> arm coefficients stay
        # unchanged across rows 0..2; row 3 has both and updates arm "b".
        c0 = out["coef"][0].to_list()
        assert out["coef"][1].to_list() == c0
        assert out["coef"][2].to_list() == c0
        assert out["coef"][3].to_list() != c0
        assert out["arm"].null_count() == 0

    def test_coef_coef_index_describe_and_min_weight(self) -> None:
        df = _frame(80, seed=10)
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1"],
            policy="linucb",
            l2=1.0,
            half_life=float("inf"),
            min_weight=10.0,
            coef_every=0,
        )
        bank = rp.ModelBank([spec])
        out = bank.fit_predict(df).unnest("b")
        assert out["arm"].slice(0, 10).null_count() == 10
        assert all(w == "below_min_weight" for w in out["withheld_reason"].slice(0, 10).to_list())
        assert out["arm"].slice(10, 70).null_count() == 0

        idx = rp.spec.coef_index(spec)
        assert idx["arm"].to_list() == [a for a in ARMS for _ in range(3)]
        assert idx["term"].to_list() == ["intercept", "x0", "x1"] * len(ARMS)
        # Check final coef vector against direct ridge on all 80 learned rows.
        last_coef = np.array(out["coef"][-1])
        X = np.column_stack([np.ones(80), df["x0"].to_numpy(), df["x1"].to_numpy()])
        a_col = np.array(df["action"].to_list())
        r_col = df["r"].to_numpy()
        for ai, a in enumerate(ARMS):
            m = a_col == a
            beta = np.linalg.solve(np.eye(3) + X[m].T @ X[m], X[m].T @ r_col[m])
            assert np.allclose(last_coef[ai * 3 : (ai + 1) * 3], beta, atol=1e-12)

        desc = bank.describe("b")
        assert desc.height >= 1

    def test_embargo_delays_learning(self) -> None:
        df = _frame(40, seed=11)
        spec = rp.spec.bandit(
            "b",
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1"],
            clock="t",
            gap_cap=10.0,
            embargo=5.0,
            half_life=50.0,
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("b")
        # First 5 rows have weight_sum == 0 because nothing has matured yet.
        assert all(w == 0.0 for w in out["weight_sum"].slice(0, 5).to_list())
        assert out["weight_sum"][-1] > 0.0

    def test_refusals(self) -> None:
        base: dict[str, Any] = dict(
            arm="action",
            arms=ARMS,
            reward="r",
            features=["x0", "x1"],
            half_life=50.0,
        )
        with pytest.raises(TypeError, match="targets"):
            rp.spec.bandit("b", targets=["r"], **base)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="alpha"):
            rp.ModelBank([rp.spec.bandit("b", policy="eps_greedy", alpha=1.0, **base)])
        with pytest.raises(ValueError, match="epsilon"):
            rp.ModelBank([rp.spec.bandit("b", policy="linucb", epsilon=0.1, **base)])
        with pytest.raises(ValueError, match="epsilon"):
            rp.ModelBank([rp.spec.bandit("b", policy="lints", epsilon=0.1, **base)])
        with pytest.raises(ValueError, match="arms"):
            rp.ModelBank([rp.spec.bandit("b", **{**base, "arms": ["only_one"]})])
        with pytest.raises(ValueError, match="arms|duplicate"):
            rp.ModelBank([rp.spec.bandit("b", **{**base, "arms": ["a", "a"]})])
        with pytest.raises(ValueError, match="arms|empty"):
            rp.ModelBank([rp.spec.bandit("b", **{**base, "arms": ["a", ""]})])
        with pytest.raises(ValueError, match="half_life|lam"):
            rp.ModelBank(
                [rp.spec.bandit("b", arm="action", arms=ARMS, reward="r", features=["x0"])]
            )
        with pytest.raises(ValueError, match="max_state_bytes"):
            rp.ModelBank([rp.spec.bandit("b", max_state_bytes=16, **base)])
        with pytest.raises(ValueError, match="min_weight"):
            rp.ModelBank([rp.spec.bandit("b", min_weight=[1.0, 2.0], **base)])
        with pytest.raises(ValueError, match="emit_target_weight"):
            rp.ModelBank([rp.spec.bandit("b", emit_target_weight=True, **base)])
        with pytest.raises(ValueError, match="emit_sigma"):
            rp.ModelBank([rp.spec.bandit("b", emit_sigma=True, **base)])

        # Unknown logged arm value at runtime names the spec, value, and row.
        bad_df = _frame(10, seed=12).with_columns(
            pl.when(pl.arange(0, 10) == 4)
            .then(pl.lit("mystery"))
            .otherwise(pl.col("action"))
            .alias("action")
        )
        bank = rp.ModelBank([rp.spec.bandit("b", **base)])
        with pytest.raises(Exception, match=r"mystery"):
            bank.fit_predict(bad_df)


# --- fifty-seed oracle grid (oracles.toml: bandit.mabwiser) -----------------

GRID_SEEDS = range(50)
GRID_ROWS = 80

GRID_FORMS: dict[str, tuple[dict[str, Any], float]] = {
    "linucb_intercept": (
        {"policy": "linucb", "alpha": 0.75, "l2": 1.0, "fit_intercept": True},
        5e-12,
    ),
    "linucb_no_intercept": (
        {"policy": "linucb", "alpha": 1.25, "l2": 1.0, "fit_intercept": False},
        5e-12,
    ),
    "eps_greedy_scores": (
        {"policy": "eps_greedy", "epsilon": 0.15, "l2": 1.0, "fit_intercept": True},
        5e-12,
    ),
}


def _grid_outputs(form: str, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    cfg, _ = GRID_FORMS[form]
    df = _frame(GRID_ROWS, 100 + seed)
    spec = rp.spec.bandit(
        "b",
        arm="action",
        arms=ARMS,
        reward="r",
        features=["x0", "x1", "x2"],
        seed=seed,
        half_life=float("inf"),
        **cfg,
    )
    out = rp.ModelBank([spec]).fit_predict(df).unnest("b")
    return df, out.select([f"score_{a}" for a in ARMS])


ledger.guard(
    "mabwiser",
    lambda: [_grid_outputs(form, seed)[1] for form in GRID_FORMS for seed in GRID_SEEDS],
    oracles=["mabwiser"],
    bits=52,
)


class TestFiftySeeds:
    """Each linear bandit form over seeds 0-49 of 80 rows against
    ``mabwiser.mab.MAB``, recorded in ``oracles.toml`` as ``bandit.mabwiser.*``."""

    @pytest.mark.oracle_grid(
        reason="50 seeds x 3 contextual bandit forms held row by row against mabwiser"
    )
    @pytest.mark.parametrize("form", list(GRID_FORMS))
    def test_fifty_seeds_against_mabwiser(self, form: str) -> None:
        cfg, floor = GRID_FORMS[form]
        errs: list[float] = []
        for seed in GRID_SEEDS:
            df, out = _grid_outputs(form, seed)
            ref = _mabwiser_scores(
                df,
                arms=ARMS,
                features=["x0", "x1", "x2"],
                policy=cfg["policy"],
                alpha=cfg.get("alpha", 1.0),
                l2=cfg["l2"],
                fit_intercept=cfg["fit_intercept"],
            )
            worst = max(float(np.max(np.abs(out[f"score_{a}"].to_numpy() - ref[a]))) for a in ARMS)
            errs.append(worst)
        ledger.record("mabwiser", form, errs, floor=floor)
        assert max(errs) <= floor, (form, max(errs))
