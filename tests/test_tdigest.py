"""The merging t-digest against the `tdigest` package and exact ranks.

PLAN 544, part 3. The core's digest (``_rheon.tdigest_quantiles``, Dunning
and Ertl's 2019 merging digest on the ``k_1`` scale) and the ``tdigest``
package (Dunning's 2013 t-digest, MIT) read the same seeded streams: normal,
lognormal, Student-t(3) and Pareto(2.5), N = 100,000, at compression 100 and
200, at the levels 0.01, 0.05, 0.5, 0.95 and 0.99.

The rank error of an estimate ``q`` at level ``p`` is ``|F(q) - p|``, ``F``
the stream's empirical CDF (the share of values at most ``q``). The two
digests are compared in rank, ``|F(q_ours) - F(q_theirs)|``, not in value:
a quantile's value moves with the distribution's scale, its rank does not.

The package's compression. ``tdigest.TDigest(delta, K)`` bounds a centroid
at quantile ``q`` to ``4 n delta q (1 - q)`` (its ``_threshold``) and
compresses once it holds more than ``K / delta`` centroids: ``delta`` is the
reciprocal of the compression (the paper's ``delta = 1 / compression``), so
compression 100 is ``delta = 0.01``, its default. It is a finer digest than
the merging one at the same compression (about 835 centroids at 100, 1,550
at 200, against the merging digest's 58 and 116), so its own rank error is
the smaller, and the difference between the two is the merging digest's.

The tolerances are the merging digest's against exact ranks, measured over
20 seeds at its worst (2.41e-3 at compression 100, 1.20e-3 at 200, at the
median) with a margin of 1.5: 3.8e-3 and 1.8e-3. The difference from the
package is held to the same, and the package's own error to the same.
Seed 0 runs in the essentials; seeds 1 to 19 in the extended tier.
"""

from __future__ import annotations

import numpy as np
import pytest
import tdigest

from rheon import _rheon as native

N = 100_000
LEVELS = (0.01, 0.05, 0.5, 0.95, 0.99)
DISTS = ("normal", "lognormal", "t3", "pareto")
TOLERANCE = {100: 3.8e-3, 200: 1.8e-3}
SEEDS = [0] + [pytest.param(s, marks=pytest.mark.extended(reason="20 seeds")) for s in range(1, 20)]


def _stream(dist: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(544_000 + seed)
    return {
        "normal": lambda: rng.normal(0.0, 1.0, N),
        "lognormal": lambda: rng.lognormal(0.0, 1.0, N),
        "t3": lambda: rng.standard_t(3, N),
        "pareto": lambda: rng.pareto(2.5, N) + 1.0,
    }[dist]()


def _ranks(xs_sorted: np.ndarray, values: list[float]) -> np.ndarray:
    return np.searchsorted(xs_sorted, np.asarray(values), side="right") / len(xs_sorted)


def rank_errors(dist: str, seed: int, compression: int) -> tuple[float, float, float]:
    """The largest, over the levels: our rank error, the package's, and
    the rank difference between the two."""
    xs = _stream(dist, seed)
    ours = native.tdigest_quantiles(xs.tolist(), float(compression), list(LEVELS))
    assert all(q is not None for q in ours)
    theirs_digest = tdigest.TDigest(delta=1.0 / compression)
    theirs_digest.batch_update(xs)
    theirs = [theirs_digest.percentile(100.0 * p) for p in LEVELS]
    s = np.sort(xs)
    f_ours = _ranks(s, ours)  # type: ignore[arg-type]
    f_theirs = _ranks(s, theirs)
    p = np.asarray(LEVELS)
    return (
        float(np.max(np.abs(f_ours - p))),
        float(np.max(np.abs(f_theirs - p))),
        float(np.max(np.abs(f_ours - f_theirs))),
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("compression", [100, 200])
@pytest.mark.parametrize("dist", DISTS)
def test_quantiles_agree_with_the_tdigest_package(dist: str, compression: int, seed: int) -> None:
    ours, theirs, between = rank_errors(dist, seed, compression)
    tol = TOLERANCE[compression]
    assert ours <= tol, f"{dist}, compression {compression}: our rank error {ours:.2e}"
    assert theirs <= tol, f"{dist}, compression {compression}: tdigest's rank error {theirs:.2e}"
    assert between <= tol, (
        f"{dist}, compression {compression}: ours and tdigest {between:.2e} apart"
    )


def test_a_coarser_digest_moves_away_from_the_package() -> None:
    """The other side of the tolerance: at compression 20 the merging digest
    (about 12 centroids) reads the median more than 3.8e-3 in rank from the
    package's digest at 100."""
    xs = _stream("normal", 0)
    (ours,) = native.tdigest_quantiles(xs.tolist(), 20.0, [0.5])
    theirs_digest = tdigest.TDigest(delta=0.01)
    theirs_digest.batch_update(xs)
    s = np.sort(xs)
    between = abs(float(_ranks(s, [ours])[0] - _ranks(s, [theirs_digest.percentile(50.0)])[0]))  # type: ignore[list-item]
    assert between > TOLERANCE[100], between
