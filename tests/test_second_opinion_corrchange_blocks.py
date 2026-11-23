"""``corrchange(kind="window", perm_block=)``: the critical value against
``scipy.stats.permutation_test`` over block numbers.

With ``perm_block = b`` the pooled rows (the two windows) are shuffled in
blocks of ``b`` consecutive rows, so with a window of ``3 b`` rows the pool is
six blocks and a draw is one of the ``C(6, 3) = 20`` ways to give three of them
to the first window. That is a permutation test over block numbers, and scipy
enumerates its exact null distribution; the model draws ``n_perm`` of its
splits at random, so its critical value (the ``(B + 1 - m)``-th smallest draw,
``m = floor(alpha (B + 1))``) is the scipy distribution's quantile once the
draws have met every split.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import permutation_test

from rheon import polars as rp

TIER = "essential"

B, NBLOCKS, K = 10, 3, 3  # the block, the blocks a window holds, the features
W = B * NBLOCKS
N_PERM = 4000


def pooled_rows() -> np.ndarray:
    """The pool: 2 W rows of three features at a correlation of 0.4, the
    second window at 0.8, so the blocks differ and the 20 splits' statistics
    are 20 distinct numbers (10 pairs, a split and its swap)."""
    rng = np.random.default_rng(550)
    f = rng.standard_normal((2 * W, 1))
    rho = np.where(np.arange(2 * W) < W, 0.4, 0.8)[:, None]
    return np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.standard_normal((2 * W, K))


def split_statistic(pool: np.ndarray):
    """The model's statistic of a split: the L1 norm of the difference of the
    two windows' correlation matrices (upper triangle), the windows given by
    block numbers."""
    iu = np.triu_indices(K, 1)

    def rows(blocks):
        return pool[np.concatenate([np.arange(b * B, (b + 1) * B) for b in blocks])]

    def statistic(first, second):
        r = [np.corrcoef(rows(s).T)[iu] for s in (first, second)]
        return float(np.abs(r[0] - r[1]).sum())

    return statistic


def model_crit(pool: np.ndarray, alpha: float, seed: int) -> float:
    cols = [f"x{i}" for i in range(K)]
    df = pl.DataFrame({c: pool[:, i] for i, c in enumerate(cols)})
    # One row more than the pool: the critical value drawn from the first
    # 2 W rows is read at row 2 W.
    df = pl.concat([df, df.tail(1)])
    spec = rp.spec.corrchange(
        "c",
        features=cols,
        kind="window",
        window_rows=W,
        alpha=alpha,
        n_perm=N_PERM,
        perm_block=B,
        seed=seed,
    )
    out = rp.ModelBank([spec]).fit_predict(df)["c"].struct.unnest()
    return float(out["crit"][2 * W])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_critical_value_is_the_exact_permutation_quantile_over_block_numbers(seed):
    """At alpha 0.15 the draw index ``q = 3401`` of 4,000 sits at 0.8503 of
    the draws, in the middle of the 9th of the ten atoms (cumulative 0.8 and
    0.9), nine standard deviations of the draws' own noise from either edge,
    so the model's critical value is scipy's exact one to rounding: a split
    and its swap have the same statistic to 1e-16."""
    pool = pooled_rows()
    stat = split_statistic(pool)
    first, second = np.arange(NBLOCKS), np.arange(NBLOCKS, 2 * NBLOCKS)
    res = permutation_test(
        (first, second),
        stat,
        permutation_type="independent",
        n_resamples=np.inf,
        alternative="greater",
    )
    null = np.sort(res.null_distribution)
    assert null.size == 20, null.size
    want = null[16]  # the 17th of 20: 0.85 of the exact distribution
    got = model_crit(pool, 0.15, seed)
    assert got == pytest.approx(want, abs=1e-12)
    # Both sides of the quantile: the atoms either side are far apart, so a
    # draw index one atom off (alpha 0.25 or 0.05) would not pass.
    assert null[14] < want - 1e-3 and null[18] > want + 1e-3, null
    assert model_crit(pool, 0.05, seed) == pytest.approx(null[-1], abs=1e-12)
    assert model_crit(pool, 0.25, seed) == pytest.approx(null[14], abs=1e-12)
